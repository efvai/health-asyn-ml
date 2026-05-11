import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml_toolbox import DataLoader, resample_dataset
from ml_toolbox import preprocessing as pp

prep_pipeline = pp.PreprocessorPipeline([
    pp.DetrendingFilter(),
    pp.ButterworthLPF(cutoff_hz=500, order=4),
])

loader = DataLoader(Path("data_set_4"))
data, meta = loader.load_batch(preprocessor=prep_pipeline)
stats = resample_dataset(data, meta, target_fs=2000.0, output_path=Path("data_set_4_2khz_float32"))
print(stats)