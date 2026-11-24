# 저사양 노트북 자원 측정 안내 (2026-11-24)

대상: Intel Core i5-1340P / DDR4 8GB / Intel Iris Xe 노트북
목적: 11/19 성능 모드 측정(기안 노트북, i7-14700HX)과 같은 조건으로 재서 논문 4.7에 저사양 결과를 추가한다.

## 11/19와 맞춰야 하는 것
- 코드·모델: 이 저장소 master 최신, `models/movenet_lightning_int8.tflite`, `models/movenet_lightning_fp16.tflite`
- 명령: 60초, 조건 base roi refine both fp16, 4회 반복
- 전원: 성능 모드(최고 성능) + 충전기 연결, 다른 프로그램 종료
- 사람: 측정 내내 한 명이 정상 자세로 앉아 있기(자리 비우면 그 회차 무효), 다른 사람이 화면에 안 들어오게
- 거리: 카메라에서 40cm 이상, 상반신이 다 보이게

장소·조명·책상·앉는 사람은 달라도 된다. 장소와 앉은 사람만 기록한다.

## 1. 설치 (처음 한 번)
1. python.org에서 Python 3.14를 설치한다("Add python.exe to PATH" 체크). 기안 노트북은 3.14.3이다.
2. PowerShell:
```
cd C:\
git clone https://github.com/gianpark/AIoT-Project.git aiot_project_repo
cd C:\aiot_project_repo
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python --version
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
python -c "import pyrealsense2, cv2, tensorflow, psutil; print('ok')"
```
- `python --version`이 3.14.x가 아니면 멈추고 알린다.
- 설치 오류가 나면 마지막 몇 줄을 공유한다.

## 2. 카메라 확인
D455를 USB 3 포트에 꽂고 아래가 `3.2`로 나오는지 본다. `2.1`이면 다른 포트에 꽂는다.
```
python -c "import pyrealsense2 as rs; d=rs.context().devices[0]; print(d.get_info(rs.camera_info.usb_type_descriptor))"
```

## 3. 측정 (약 25분)
새 PowerShell을 열었다면 먼저 `cd C:\aiot_project_repo` 와 `.venv\Scripts\Activate.ps1`.
```
foreach ($i in 1..4) { python -m scripts.measure_resources --seconds 60 --conditions base roi refine both fp16 --out experiments/resource_lowspec_run$i.md }
python -m scripts.summarize_resources experiments/resource_lowspec_run1.md experiments/resource_lowspec_run2.md experiments/resource_lowspec_run3.md experiments/resource_lowspec_run4.md --out experiments/resource_lowspec_summary.md
```
- 한 회차는 조건 5개 × 70초(앞 10초는 통계 제외)라 약 6분이다.
- 측정 중 창을 닫거나 q를 누르지 않는다.

## 4. 사양 기록
측정 결과 파일의 하드웨어 줄은 CPU 이름이 정확히 안 나오므로 따로 기록한다. 메모리 줄 개수 = 꽂힌 RAM 개수.
```
Get-CimInstance Win32_Processor | Select-Object Name
Get-CimInstance Win32_PhysicalMemory | Select-Object Capacity, Speed
python --version
```

## 기록란
- 날짜·시간:
- 노트북 모델명:
- 전원 모드 / 충전기:
- RAM 개수(싱글/듀얼 채널):
- 장소 / 앉은 사람:
- 자리 비운 시각(있으면):
- 이상 현상:

## 끝나면
`experiments/resource_lowspec_run1~4.md`, `experiments/resource_lowspec_summary.md` 5개와 기록란을 기안에게 보낸다.
