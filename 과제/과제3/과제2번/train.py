import os
import argparse
from solver import Solver

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_epochs", type=int, default=100)
    
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoint") #모델을 하면서 세입할때 
    parser.add_argument("--ckpt_name", type=str, default="flower")#파일이름을 뭐로할껀가
    parser.add_argument("--print_every", type=int, default=1)
   
    
    # if you change image size, you must change all the network channels
    parser.add_argument("--image_size", type=int, default=64) #이미지크기를 정하는거  이미지사이즈 = 64* 64* 3이다 이제 
    parser.add_argument("--data_root", type=str, default="./data")

    args = parser.parse_args()
    solver = Solver(args)# solver는 학습만하는 class로만들어놧다 
    solver.fit() #트레이닝하는 함수

if __name__ == "__main__":
    main()
