import os
import json
import urllib.request
import time

def main():
    base_dir = "/gpfs/data/bbj-lab/users/daniel/input-representation-benchmark"
    lit_dir = os.path.join(base_dir, "literature")
    json_path = os.path.join(base_dir, "methods", "arxiv_sources.json")
    
    os.makedirs(lit_dir, exist_ok=True)
    
    with open(json_path, 'r') as f:
        sources = json.load(f)
        
    for source in sources:
        key = source.get("key")
        arxiv_id = source.get("arxiv_id")
        if not key or not arxiv_id:
            continue
            
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        out_path = os.path.join(lit_dir, f"{key}.pdf")
        
        if os.path.exists(out_path):
            print(f"Already downloaded: {out_path}")
            continue
            
        print(f"Downloading {arxiv_id} for {key}...")
        try:
            # Need a user agent because some servers block standard urllib UA
            req = urllib.request.Request(pdf_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(out_path, 'wb') as out_file:
                out_file.write(response.read())
            print(f"Saved to {out_path}")
            time.sleep(1) # Be nice to arxiv
        except Exception as e:
            print(f"Failed to download {pdf_url}: {e}")

if __name__ == "__main__":
    main()
