# kipris

데이터 전처리
- 불필요한 특정문자 선별 삭제 : “\‘삭제\” , “\’삭제\‘”, “\\”, “\’,”, “\‘” 차례로 삭제, “None” 삭제
- 쉼표 -> 공백문자로 변경
- 관용적인 표현 삭제 : “제 X항에 있어서”와 같은 문장 삭제
- 순번을 지정하는 문자 삭제 : “(710 및 720)”, “(a6 및 g6)”, “(100 및 101)” 등 삭제
- 한글을 제외한 문자 모두 삭제
- 공백 2개이상 -> 1개로 변경
- 정제로 인한 중복문자 제거

명령어 예시
- CUDA_VISIBLE_DEVICES=0 python train.py -Level 3 -s_year 2003 -e_year 2017 -batch_size 64 -max_length 256 -kisti_label false -desc all_Batch_Size_64_Max_Len_256
- Level : 데이터 파일 이름 (전처리 level)
- s_year : 데이터 파일 이름 (시작 연도)
- e_year : 데이터 파일 이름 (끝 연도)
- batch_size : 배치 사이즈
- max_length : 최대 시퀀스 길이
- kisti_label : kisti 데이터 label만 사용할지 유무
- desc : 저장될 파일명 (추가 설명을 작성)

