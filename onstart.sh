export HF_HOME="/workspace/.cache/huggingface"

apt update

python -m pip install --upgrade pip

pip install -r requirements.txt

huggingface-cli download microsoft/Phi-3-mini-4k-instruct

cd LAVIS 
python convert_hf_model.py

cd ..
wget -c http://images.cocodataset.org/zips/train2017.zip

apt install p7zip-full

7z x train2017.zip

