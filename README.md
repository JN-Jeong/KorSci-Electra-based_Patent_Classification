# electra_JN
kipris

명령어 예시
-Level : 데이터 파일 이름 (전처리 level)
-s_year : 데이터 파일 이름 (시작 연도)
-e_year : 데이터 파일 이름 (끝 연도)
-batch_size : 배치 사이즈
-max_length : 최대 시퀀스 길이
-kisti_label : kisti 데이터 label만 사용할지 유무
-desc : 저장될 파일명 (추가 설명을 작성)
CUDA_VISIBLE_DEVICES=0 python train.py -Level 3 -s_year 2003 -e_year 2017 -batch_size 64 -max_length 256 -kisti_label false -desc all_Batch_Size_64_Max_Len_256
