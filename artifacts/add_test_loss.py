from pathlib import Path
p=Path('reproduction/reproduce.py');s=p.read_text().replace("test_top5=ev['top5'],test_n=ev['n']", "test_top5=ev['top5'],test_ce=ev['ce'],test_n=ev['n']");p.write_text(s)
