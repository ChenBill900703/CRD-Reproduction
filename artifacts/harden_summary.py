from pathlib import Path
p=Path('reproduction/summarize.py');s=p.read_text().replace("   valid=sorted(seeds)==[0,1,2,3,4]", "   valid=sorted(seeds)==[0,1,2,3,4] and len({r['teacher_sha256'] for r in selected})==1")
p.write_text(s)
