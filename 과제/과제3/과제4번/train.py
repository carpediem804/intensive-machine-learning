import os
import argparse
from solver import Solver

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001) ##learning late 설정
    parser.add_argument("--batch_size", type=int, default=128) ## batch_size = 128
    parser.add_argument("--max_epochs", type=int, default=200) # epoch를 200으로 
    parser.add_argument("--bptt_len", type=int, default=30) ##30개씩 잘라서 미분할때 함 
    
    parser.add_argument("--embed_dim", type=int, default=300) # embedding할때 차원정해주기
    parser.add_argument("--hidden_dim", type=int, default=512) #hidden layer의 차원정해주기
    parser.add_argument("--num_layers", type=int, default=3) #레이어 갯수 =3
    
    parser.add_argument("--ckpt_dir", type=str, default="checkpoint") #체크포인트 디렉토리
    parser.add_argument("--ckpt_name", type=str, default="char-rnn") 
    parser.add_argument("--print_every", type=int, default=1)
    
    parser.add_argument("--result_dir", type=str, default="result") # 결과디렉토리

    args = parser.parse_args()
    solver = Solver(args)
    solver.fit()

if __name__ == "__main__":
    main()
