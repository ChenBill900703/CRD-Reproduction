from pathlib import Path
p=Path('artifacts/download_data.py'); s=p.read_text().replace('def fetch(i):','def fetch_once(i):').replace(" req=urllib.request.Request", " if (root/str(i)).exists() and (root/str(i)).stat().st_size==end-start+1: return\n req=urllib.request.Request"); s=s.replace('with concurrent.futures.ThreadPoolExecutor(max_workers=n)', '''def fetch(i):
 for attempt in range(5):
  try: return fetch_once(i)
  except Exception as e:
   print('retry',i,attempt,repr(e),flush=True)
   if attempt==4: raise
with concurrent.futures.ThreadPoolExecutor(max_workers=4)''');p.write_text(s)
