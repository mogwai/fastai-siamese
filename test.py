from siamese import *
import sys
sys.path.append("../audio")
from audio import *
from siamese import *
np.random.seed(2)

data_url = 'http://www.openslr.org/resources/45/ST-AEDS-20180100_1-OS'
data_folder = datapath4file(url2name(data_url))
untar_data(data_url, dest=data_folder)
label_pattern = r'_/([mf]\d+)_'
config = AudioTransformConfig()
config.segment_size = 1000
# config.remove_silence = True
config.f_max = 8000
config.to_db_scale = True
config.top_db = 80
# config.silence_threshold = 30
config.max_to_pad = 1000
# config.silence_padding = 300
audios = AudioList.from_folder(data_folder, config=config).split_none().label_from_re(label_pattern)
sds = SiameseDataset.create_from_ll(audios, split_c=.2, tar_num=10000)
ex = sds.train[0]