from pathlib import Path
import requests

MDX_DOWNLOAD_LINK = 'https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/'
RVC_DOWNLOAD_LINK = 'https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/'
PRETRAINED_V1_DOWNLOAD_LINK = 'https://huggingface.co/Salmizu/Pretrained/resolve/main/'

BASE_DIR = Path(__file__).resolve().parent.parent
mdxnet_models_dir = BASE_DIR / 'mdxnet_models'
rvc_models_dir = BASE_DIR / 'rvc_models'
pretrained_v1_models_dir = BASE_DIR / 'pretrained'


def dl_model(link, model_name, dir_name):
    with requests.get(f'{link}{model_name}') as r:
        r.raise_for_status()
        with open(dir_name / model_name, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


if __name__ == '__main__':
    mdx_model_names = ['UVR-MDX-NET-Voc_FT.onnx', 'UVR_MDXNET_KARA_2.onnx', 'Reverb_HQ_By_FoxJoy.onnx']
    for model in mdx_model_names:
        print(f'Downloading {model}...')
        dl_model(MDX_DOWNLOAD_LINK, model, mdxnet_models_dir)

    rvc_model_names = ['hubert_base.pt', 'rmvpe.pt']
    for model in rvc_model_names:
        print(f'Downloading {model}...')
        dl_model(RVC_DOWNLOAD_LINK, model, rvc_models_dir)

    pretrained_v1_model_names = ['D32k.pth', 'D42k.pth', 'D48k.pth', 'G32k.pth', 'G42k.pth', 'G48k.pth', 'f0D32k.pth', 'f0D42k.pth' 'f0D48k.pth', 'f0G32k.pth', 'f0G42k.pth, 'f0G48k.pth']
    for model in pretrained_v1_model_name:
        print(f'Downloading {model}...'
        dl_model(PRETRAINED_V1_DOWNLOAD_LINK, model, pretrained_v1_models_dir

    print('All models downloaded!')
