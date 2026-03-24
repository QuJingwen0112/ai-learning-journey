import random
num=random.randint(1,10)
while True:
    guess=int(input("请输入："))
    if guess==num:
        print("Great")
    elif guess<num:
        print("It's too small")
    else:
        print("It's too large")
