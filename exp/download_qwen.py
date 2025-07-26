import os

# 使用ModelScope国内源下载Qwen2.5-7B-Instruct-1M
# 需先安装modelscope和transformers
# pip install modelscope transformers

def download_qwen_model():
    from modelscope.hub.snapshot_download import snapshot_download
    model_id = 'Qwen/Qwen2.5-7B-Instruct-1M'
    # 指定国内源
    model_dir = snapshot_download(model_id, cache_dir='./models', revision=None, local_files_only=False)
    print(f"模型已下载到: {model_dir}")

if __name__ == '__main__':
    download_qwen_model() 