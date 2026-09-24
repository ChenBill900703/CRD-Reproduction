from pathlib import Path
p=Path('reproduction/reproduce.py');s=p.read_text().replace("    if c.get('profile') not in", "    allowed=set(fixed)|set(extra_fixed)|{'profile','method','epochs','batch_size','nce_k','nce_t','loss_weights','max_steps','eval_steps','seed','trial','data','teacher','download','evaluate_teacher'}\n    unknown=set(c)-allowed\n    if unknown: raise ValueError(f'Unknown config fields: {sorted(unknown)}')\n    if c.get('profile') not in")
s=s.replace("    if 'trial' in c and", "    if c['profile']!='smoke' and 'seed' in c and c['seed'] not in c['seed_list']: raise ValueError('Formal seed not in preregistered list')\n    if 'trial' in c and")
p.write_text(s)
p=Path('reproduction/test_preflight.py');s=p.read_text().replace("[('profile','typo')", "[('warmup',True),('profile','typo')");p.write_text(s)
