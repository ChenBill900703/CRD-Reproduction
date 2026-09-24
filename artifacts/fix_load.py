from pathlib import Path
p=Path('reproduction/reproduce.py'); s=p.read_text().replace("ck=torch.load(c['teacher'],map_location='cpu',weights_only=True)","with torch.serialization.safe_globals([(np._core.multiarray.scalar, 'numpy.core.multiarray.scalar'), np.dtype, np.dtypes.Float64DType]):\n            ck=torch.load(c['teacher'],map_location='cpu',weights_only=True)"); p.write_text(s)
