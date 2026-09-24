import json,tempfile,unittest
from pathlib import Path
from summarize import collect,summarize,ROOT,TEACHER_SHA256
class Results(unittest.TestCase):
    def fixture(self,root):
        for seed in range(5):
            p=root/str(seed);p.mkdir();c=json.loads((ROOT/'configs/author_code_crd.json').read_text());c.update(seed=seed,trial='999')
            (p/'config.json').write_text(json.dumps(c));(p/'environment.json').write_text(json.dumps({'tested':'same'}))
            (p/'status.json').write_text(json.dumps(dict(status='complete',epoch=240,global_step=782*240,train_n=50000,test_n=10000,code_fingerprint='same',final_top1=70+seed,best_top1=99,teacher={'sha256':TEACHER_SHA256,'n':10000})))
    def test_final_not_best_and_seed_not_trial(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root);rows=collect(root);result=summarize(rows)[0]
            self.assertEqual(result['mean'],72);self.assertTrue(result['formal_five_seed']);self.assertEqual(rows[0]['trial'],'999')
            self.assertAlmostEqual(result['std_sample_ddof1'],(2.5)**.5)
            status=root/'0/status.json';q=json.loads(status.read_text());q['epoch']=239;status.write_text(json.dumps(q))
            self.assertFalse(summarize(collect(root))[0]['formal_five_seed'])
    def test_duplicate_seed_and_mixed_code_not_formal(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root);p=root/'0/config.json';c=json.loads(p.read_text());c['seed']=1;p.write_text(json.dumps(c))
            self.assertFalse(summarize(collect(root))[0]['formal_five_seed'])
            p=root/'0/status.json';s=json.loads(p.read_text());s['code_fingerprint']='changed';p.write_text(json.dumps(s))
            result=summarize(collect(root))[0];self.assertFalse(result['consistent']);self.assertIsNone(result['mean'])
    def test_invalid_final_teacher_and_missing_files_preserved(self):
        for field,value in [('final_top1',None),('final_top1',float('nan')),('global_step',1),('test_n',64),('teacher',{})]:
            with self.subTest(field=field),tempfile.TemporaryDirectory() as d:
                root=Path(d);self.fixture(root);p=root/'0/status.json';s=json.loads(p.read_text());s[field]=value;p.write_text(json.dumps(s))
                self.assertFalse(collect(root)[0]['eligible'])
                (root/'broken').mkdir();self.assertEqual(collect(root)[-1]['status'],'unreadable')
if __name__=='__main__':unittest.main(verbosity=2)
