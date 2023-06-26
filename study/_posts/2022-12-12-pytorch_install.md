---
layout: post
title: pytorch_install
description: >
    파이토치 설치
sitemap: false
hide_last_modified: true
---

공식 문서에 나와있는 cuda 11.6 설치 방법
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

설치하면
torch 1.10.1
cuda 102가 설치됨

python 버전에 맞게 추가해서 설치
-> python3.8 -m pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116

1) ModuleNotFoundError: No module named 'setuptools' 에러 발생
: python3.8 -m pip install --upgrade pip setuptools wheel
: pip3 install --upgrade pip setuptools


원인
import sys
sys.version
주피터에서 위의 코드를 실행해보면 python3.8이 아닌 python3.6.9가 나오는 것을 알 수 있다. 커널에서 torch를 uninstall해도 깔아둔 3.8에서 uninstall될 뿐 3.6.9에서는 그대로 install되어있다.

jupyter notebook에서 kernel을 추가하는 것으로 해결
python3.7 -m pip insatll ipykernel
python3.7 -m ipykernel install --user --name="Python3.8"
-> 띄어쓰기는 이름에 사용 불가능하다
