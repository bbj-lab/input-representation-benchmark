"""Quantization with 20-bin ventile support and reference range handling.

This module extends the quantization approach from ethos-ares to support:
1. Ventile (20-bin) quantization instead of decile (10-bin)
2. Reference range-aware binning using ref_range_lower and ref_range_upper
3. Out-of-range bin allocation (5 bins below lower, 5 bins above upper)
4. Flexible binning strategies based on available reference ranges
"""

import json
from pathlib import Path

import numpy as np
import polars as pl


def static_class(cls):
    """Decorator to instantiate a class immediately."""
    return cls()


@static_class
class QuantizationVentile:
    """Quantizator for lab values using 20-bin ventile with reference range support.
    
    Binning strategies:
    - Both ref ranges: 10 bins within range + 5 below + 5 above = 20 bins
    - One ref range: 15 bins within range + 5 outside = 20 bins  
    - No ref ranges: 20 bins (ventile) across data distribution
    """
    
    def __call__(
        self, 
        df: pl.DataFrame, 
        *, 
        code_prefixes: list[str],
        include_ref_ranges: bool = True
    ) -> pl.DataFrame:
        """Filter and aggregate numeric values with optional reference ranges.
        
        Args:
            df: DataFrame with code, numeric_value columns
            code_prefixes: List of code prefixes to filter (e.g., ["LAB//Q//"])
            include_ref_ranges: Whether to include ref_range columns if present
            
        Returns:
            Aggregated DataFrame with code and numeric_value (and ref ranges if requested)
        """
        expr = self._create_prefix_or_chain(code_prefixes)
        filtered_df = df.filter(expr)
        
        agg_cols = ["numeric_value"]
        if include_ref_ranges and "ref_range_lower" in df.columns:
            agg_cols.extend(["ref_range_lower", "ref_range_upper"])
        
        return filtered_df.group_by("code").agg(agg_cols)
    
    def agg(
        self,
        in_fps: list[str | Path],
        out_fp: str | Path,
        *,
        num_quantiles: int = 20,
        code_prefixes: list[str] | None = None,
    ) -> None:
        """Aggregate quantile breaks from multiple files with reference range support.
        
        This generates ventile (20-bin) breaks per code, respecting reference ranges:
        - If both ref_range_lower and ref_range_upper exist: 5 below + 10 within + 5 above
        - If no ref ranges: 20 bins (ventile) across entire distribution
        
        Args:
            in_fps: List of input parquet file paths
            out_fp: Output JSON file path for quantile breaks
            num_quantiles: Number of bins (default 20 for ventile)
        """
        # Load all files and concatenate (simpler than join)
        # Check if ref_range columns exist
        first_schema = pl.scan_parquet(in_fps[0]).collect_schema()
        has_ref_lower = "ref_range_lower" in first_schema.names()
        has_ref_upper = "ref_range_upper" in first_schema.names()
        
        # Load all files
        select_cols = ["code", "numeric_value"]
        if has_ref_lower:
            select_cols.append("ref_range_lower")
        if has_ref_upper:
            select_cols.append("ref_range_upper")
        
        dfs = [pl.scan_parquet(fp).select(select_cols) for fp in in_fps]
        
        # Concatenate all files
        combined = pl.concat(dfs)
        
        # Filter to only specified code prefixes if provided
        if code_prefixes:
            expr = self._create_prefix_or_chain(code_prefixes)
            combined = combined.filter(expr)
        
        # Group by code and aggregate
        agg_exprs = [pl.col("numeric_value")]
        if has_ref_lower:
            agg_exprs.append(pl.col("ref_range_lower").drop_nulls().first().alias("ref_range_lower"))
        if has_ref_upper:
            agg_exprs.append(pl.col("ref_range_upper").drop_nulls().first().alias("ref_range_upper"))
        
        collected_df = combined.group_by("code").agg(agg_exprs).collect()
        
        # Generate quantile breaks for each code
        quantile_dict = {}
        
        for row in collected_df.iter_rows(named=True):
            code = row["code"]
            values = row["numeric_value"]
            
            if not values or len(values) == 0:
                continue
            
            ref_lower = row.get("ref_range_lower") if has_ref_lower else None
            ref_upper = row.get("ref_range_upper") if has_ref_upper else None
            
            breaks = self._compute_ventile_breaks(
                values, 
                ref_lower, 
                ref_upper, 
                num_quantiles
            )
            
            if breaks:
                quantile_dict[code] = breaks
        
        with Path(out_fp).open("w") as f:
            json.dump(quantile_dict, f)
    
    def _compute_ventile_breaks(
        self,
        values: list[float],
        ref_lower: float | None,
        ref_upper: float | None,
        num_bins: int = 20,
    ) -> list[float]:
        """Compute ventile breaks respecting reference ranges.
        
        Binning strategy (simplified based on MIMIC-IV v3.1 analysis):
        - Both ranges: 5 bins below + 10 bins within + 5 bins above = 20 bins
        - No ranges: 20 bins (ventile) across data
        
        Note: In MIMIC-IV v3.1, reference ranges are always paired (both or neither).
        Lower-only and upper-only strategies are not needed (0% occurrence).
        
        Args:
            values: List of numeric values
            ref_lower: Reference range lower bound (optional)
            ref_upper: Reference range upper bound (optional)
            num_bins: Total number of bins (default 20)
            
        Returns:
            List of break points (always num_bins - 1 breaks for consistency)
        """
        values_array = np.array([v for v in values if v is not None and not np.isnan(v)])
        
        if len(values_array) == 0:
            return []
            
        # Strict filtering: If unique values < num_bins, do not use this code
        if len(np.unique(values_array)) < num_bins:
            return []
        
        # Simplified: Only 2 cases based on MIMIC-IV v3.1 evidence
        if ref_lower is not None and ref_upper is not None and ref_lower < ref_upper:
            # Case 1: Both ranges (80.3% of events)
            breaks = self._compute_breaks_both_ranges(
                values_array, ref_lower, ref_upper, num_bins
            )
        else:
            # Case 2: No ranges (19.7% of events)
            breaks = self._compute_breaks_no_ranges(values_array, num_bins)
        
        # Return breaks without deduplication to guarantee num_bins
        return breaks
    
    def _compute_breaks_both_ranges(
        self,
        values: np.ndarray,
        ref_lower: float,
        ref_upper: float,
        num_bins: int = 20,
    ) -> list[float]:
        """Compute breaks with both reference ranges: 5 below + 10 within + 5 above = 20 bins.
        
        Strict filtering: Returns empty list if we cannot achieve the target bins in all regions.
        """
        breaks = []
        
        # 5 bins below reference range (4 internal breaks) - requires 5 unique values
        below_values = values[values < ref_lower]
        unique_below = len(np.unique(below_values)) if len(below_values) > 0 else 0
        
        # 10 bins within reference range (9 internal breaks) - requires 10 unique values
        within_values = values[(values >= ref_lower) & (values <= ref_upper)]
        unique_within = len(np.unique(within_values)) if len(within_values) > 0 else 0
        
        # 5 bins above reference range (4 internal breaks) - requires 5 unique values
        above_values = values[values > ref_upper]
        unique_above = len(np.unique(above_values)) if len(above_values) > 0 else 0
        
        # Strict filtering: Each region must have enough unique values for the required bins
        # We need at least 5 unique values below, 10 within, and 5 above
        if unique_below < 5 or unique_within < 10 or unique_above < 5:
            return []  # Reject this code - insufficient data for 5-10-5 binning
        
        # Compute breaks for below region
        if len(below_values) >= 5:
            below_quantiles = np.linspace(0, 1, 6)[1:-1]  # 4 internal breaks
            below_breaks = np.quantile(below_values, below_quantiles)
            breaks.extend(below_breaks.tolist())
        
        # Reference lower bound is a break point
        breaks.append(float(ref_lower))
        
        # Compute breaks for within region
        if len(within_values) >= 10:
            within_quantiles = np.linspace(0, 1, 11)[1:-1]  # 9 internal breaks
            within_breaks = np.quantile(within_values, within_quantiles)
            breaks.extend(within_breaks.tolist())
        
        # Reference upper bound is a break point
        breaks.append(float(ref_upper))
        
        # Compute breaks for above region
        if len(above_values) >= 5:
            above_quantiles = np.linspace(0, 1, 6)[1:-1]  # 4 internal breaks
            above_breaks = np.quantile(above_values, above_quantiles)
            breaks.extend(above_breaks.tolist())
        
        # Ensure exactly 19 breaks (20 bins)
        # Total: 4 (below) + 1 (ref_lower) + 9 (within) + 1 (ref_upper) + 4 (above) = 19
        return sorted(breaks)
    
    
    def _compute_breaks_no_ranges(
        self,
        values: np.ndarray,
        num_bins: int = 20,
    ) -> list[float]:
        """Compute breaks without reference ranges: 20 bins (ventile) across data.
        
        Always returns exactly num_bins - 1 breaks for num_bins bins.
        """
        if len(values) < num_bins:
            # If fewer values than bins, use all unique values as breaks
            unique_vals = np.unique(values)
            if len(unique_vals) <= 1:
                return []
            return sorted(unique_vals[:-1].tolist())
        
        quantiles = np.linspace(0, 1, num_bins + 1)[1:-1]  # num_bins - 1 breaks
        breaks = np.quantile(values, quantiles)
        return breaks.tolist()
    
    @staticmethod
    def _create_prefix_or_chain(prefixes: list[str]) -> pl.Expr:
        """Create polars expression to filter by code prefixes."""
        expr = pl.lit(False)
        for prefix in prefixes:
            expr = expr | pl.col("code").str.starts_with(prefix)
        return expr


def transform_to_ventiles(
    df: pl.DataFrame, 
    *, 
    code_quantiles: str | Path | dict
) -> pl.DataFrame:
    """Transform numeric values in the DataFrame to ventile bins based on provided breaks.
    
    This function maps numeric values to quantile bins (V1-V20) using pre-computed
    break points that respect reference ranges.
    
    Args:
        df: DataFrame with 'code' and 'numeric_value' columns
        code_quantiles: Path to JSON file or dict mapping codes to quantile breaks
        
    Returns:
        DataFrame with codes transformed to ventile bins (e.g., "V1", "V2", ..., "V20")
        
    Examples:
        Basic usage with ventile breaks:
        >>> df = pl.DataFrame({
        ...     "code": ["A", "A", "A", "B", "B"],
        ...     "numeric_value": [1.0, 5.0, 10.0, 20.0, 30.0]
        ... })
        >>> code_quantiles = {
        ...     "A": [2.0, 7.0],  # 3 bins
        ...     "B": [25.0],      # 2 bins
        ... }
        >>> transform_to_ventiles(df, code_quantiles=code_quantiles)
        shape: (5, 2)
        ┌──────┬───────────────┐
        │ code ┆ numeric_value │
        │ ---  ┆ ---           │
        │ str  ┆ f64           │
        ╞══════╪═══════════════╡
        │ V1   ┆ 1.0           │
        │ V2   ┆ 5.0           │
        │ V3   ┆ 10.0          │
        │ V1   ┆ 20.0          │
        │ V2   ┆ 30.0          │
        └──────┴───────────────┘
    """
    if not isinstance(code_quantiles, dict):
        with Path(code_quantiles).open("r") as f:
            code_quantiles = json.load(f)
    
    # Find maximum number of breaks to determine temp column count
    max_length = max(len(v) for v in code_quantiles.values()) if code_quantiles else 0
    tmp_cols = [f"field_{i}" for i in range(max_length)]
    
    return (
        df.with_columns(
            pl.col("code")
            .replace_strict(code_quantiles, default=None, return_dtype=pl.List(pl.Float64))
            .list.to_struct(fields=tmp_cols)
            .struct.unnest()
        )
        .with_columns(
            code=pl.when(pl.col(tmp_cols[0]).is_not_null())
            .then(
                "V"  # Use "V" for Ventile instead of "Q" for Quantile
                + (
                    pl.sum_horizontal(
                        pl.col(tmp_cols) < pl.col("numeric_value"),
                    ) + 1
                ).cast(pl.String)
            )
            .otherwise(pl.col("code"))
        )
        .drop(tmp_cols)
    )

