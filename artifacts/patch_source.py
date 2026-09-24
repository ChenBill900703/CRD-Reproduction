from pathlib import Path
p=Path('reproduction/dataset/cifar100.py')
s=p.read_text(); s=s.replace('self.train_data','self.data').replace('self.test_data','self.data').replace('self.train_labels','self.targets').replace('self.test_labels','self.targets'); p.write_text(s)
p=Path('reproduction/crd/memory.py'); s=p.read_text().replace('        self.multinomial.cuda()\n',''); s=s.replace('            idx = self.multinomial.draw', '            self.multinomial.prob = self.multinomial.prob.to(v1.device)\n            self.multinomial.alias = self.multinomial.alias.to(v1.device)\n            idx = self.multinomial.draw'); p.write_text(s)
p=Path('reproduction/distiller_zoo/KD.py'); s=p.read_text().replace('size_average=False','reduction="sum"'); p.write_text(s)
