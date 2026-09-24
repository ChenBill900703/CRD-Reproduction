from pathlib import Path
p=Path('reproduction/test_preflight.py');s=p.read_text();marker="    def test_full_crd_upstream_parity_both_temperatures(self):";s=s.replace(marker,'''    def test_plot_synthetic_metrics(self):
        from plot_metrics import plot_runs
        root=BASE/'artifacts/preflight/plot-fixture';folder=root/'synthetic_fixture_NOT_RESEARCH_RESULTS';folder.mkdir(parents=True,exist_ok=True)
        (folder/'config.json').write_text(json.dumps({'max_steps':1}))
        rows=[dict(epoch=i,ce=4.6,kd=.2,crd=10.,total=12.8,train_top1=1.,test_top1=2.,test_top5=5.,lr=.05) for i in (1,2)]
        (folder/'metrics.jsonl').write_text(''.join(json.dumps(row)+'\\n' for row in rows))
        paths=plot_runs(root);self.assertEqual(len(paths),1)
        from PIL import Image
        with Image.open(paths[0]) as im:self.assertEqual(im.size,(1950,525))

'''+marker);p.write_text(s)
