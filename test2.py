# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 13:03:50 2018

@author: lj
"""
import numpy as np
import pandas as pd
from sklearn import model_selection
import random
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor


## GA优化算法
class GA(object):
    ###1.1 初始化
    def __init__(self, population_size, chromosome_num, chromosome_length, max_value, iter_num, pc, pm,
                 X_train,X_test, y_train, y_test):
        '''初始化参数
        input:population_size(int):种群数
              chromosome_num(int):染色体数，对应需要寻优的参数个数
              chromosome_length:染色体的基因长度
              max_value(float):作用于二进制基因转化为染色体十进制数值
              iter_num(int):迭代次数
              pc(float):交叉概率阈值(0<pc<1)
              pm(float):变异概率阈值(0<pm<1)
        '''
        self.population_size = population_size
        self.choromosome_length = chromosome_length
        self.chromosome_num = chromosome_num
        self.iter_num = iter_num
        self.max_value = max_value
        self.pc = pc  ##一般取值0.4~0.99
        self.pm = pm  ##一般取值0.0001~0.1

        self.X_train=X_train
        self.X_test=X_test
        self.y_train=y_train
        self.y_test=y_test


    def species_origin(self):
        '''初始化种群、染色体、基因
        input:self(object):定义的类参数
        output:population(list):种群
        '''
        population = []
        ## 分别初始化两个染色体
        for i in range(self.chromosome_num):
            tmp1 = []  ##暂存器1，用于暂存一个染色体的全部可能二进制基因取值
            for j in range(self.population_size):
                tmp2 = []  ##暂存器2，用于暂存一个染色体的基因的每一位二进制取值
                for l in range(self.choromosome_length):
                    tmp2.append(random.randint(0, 1))  # 随机产生0，1
                tmp1.append(tmp2)
            population.append(tmp1)
        return population

    ###1.2 计算适应度函数值
    def translation(self, population):
        '''将染色体的二进制基因转换为十进制取值
        input:self(object):定义的类参数
              population(list):种群
        output:population_decimalism(list):种群每个染色体取值的十进制数
        '''
        population_decimalism = []
        for i in range(len(population)):
            tmp = []  ##暂存器，用于暂存一个染色体的全部可能十进制取值
            for j in range(len(population[0])):
                total = 0.0
                for l in range(len(population[0][0])):
                    total += population[i][j][l] * (math.pow(2, l))
                tmp.append(total)
            population_decimalism.append(tmp)
        return population_decimalism

    def fitness(self, population):
        '''计算每一组染色体对应的适应度函数值
        input:self(object):定义的类参数
              population(list):种群
        output:fitness_value(list):每一组染色体对应的适应度函数值
        '''
        fitness = []
        population_decimalism = self.translation(population)  # 将2进制数据翻译为10进制数据
        for i in range(len(population[0])):  # len(population[0])=270
            tmp = []  ##暂存器，用于暂存每组染色体十进制数值，共chromosome_num=2组，每组有population_size=270个决策变量
            for j in range(2):  # len(population)
                value = population_decimalism[j][i] / (math.pow(2, self.choromosome_length) - 1)

                tmp.append(value)

            canshu_1 = [8000, 10000]
            canshu_2 = [1, 10]

            cs_1 = tmp[0] * (canshu_1[1] - canshu_1[0]) + canshu_1[0]
            cs_2 = tmp[1] * (canshu_2[1] - canshu_2[0]) + canshu_2[0]

            model = GradientBoostingRegressor(random_state=123, n_estimators=int(cs_1 // 100),
                                              learning_rate=abs(round(cs_2 * 0.001, 3)))
            model.fit(self.X_train, self.y_train)

            score = model.score(self.X_test, self.y_test)
            fitness.append(score)

        ##将适应度函数值中为负数的数值排除
        fitness_value = []
        num = len(fitness)
        for l in range(num):
            if (fitness[l] > 0):
                tmp1 = fitness[l]
            else:
                tmp1 = 0.0
            fitness_value.append(tmp1)
        return fitness_value

    ###1.3 选择操作
    def sum_value(self, fitness_value):
        '''适应度求和
        input:self(object):定义的类参数
              fitness_value(list):每组染色体对应的适应度函数值
        output:total(float):适应度函数值之和
        '''
        total = 0.0
        for i in range(len(fitness_value)):
            total += fitness_value[i]
        return total

    def cumsum(self, fitness1):
        '''计算适应度函数值累加列表
        input:self(object):定义的类参数
              fitness1(list):适应度函数值列表
        output:适应度函数值累加列表
        '''
        ##计算适应度函数值累加列表
        for i in range(len(fitness1) - 1, -1, -1):  # range(start,stop,[step]) # 倒计数
            total = 0.0
            j = 0
            while (j <= i):
                total += fitness1[j]
                j += 1
            fitness1[i] = total

    def selection(self, population, fitness_value):
        '''选择操作
        input:self(object):定义的类参数
              population(list):当前种群
              fitness_value(list):每一组染色体对应的适应度函数值
        '''
        new_fitness = []  ## 用于存储适应度函归一化数值
        total_fitness = self.sum_value(fitness_value)  ## 适应度函数值之和
        for i in range(len(fitness_value)):
            new_fitness.append(fitness_value[i] / total_fitness)

        self.cumsum(new_fitness)

        ms = []  ##用于存档随机数
        pop_len = len(population[0])  ##种群数

        for i in range(pop_len):
            ms.append(random.randint(0, 1))
        ms.sort()  ## 随机数从小到大排列

        ##存储每个染色体的取值指针
        fitin = 0
        newin = 0

        new_population = population

        ## 轮盘赌方式选择染色体
        while newin < pop_len & fitin < pop_len:
            if (ms[newin] < new_fitness[fitin]):
                for j in range(len(population)):
                    new_population[j][newin] = population[j][fitin]
                newin += 1
            else:
                fitin += 1

        population = new_population

    ### 1.4 交叉操作
    def crossover(self, population):
        '''交叉操作
        input:self(object):定义的类参数
              population(list):当前种群
        '''
        pop_len = len(population[0])

        for i in range(len(population)):
            for j in range(pop_len - 1):
                if (random.random() < self.pc):
                    cpoint = random.randint(0, len(population[i][j]))  ## 随机选择基因中的交叉点
                    ###实现相邻的染色体基因取值的交叉
                    tmp1 = []
                    tmp2 = []
                    # 将tmp1作为暂存器，暂时存放第i个染色体第j个取值中的前0到cpoint个基因，
                    # 然后再把第i个染色体第j+1个取值中的后面的基因，补充到tem1后面
                    tmp1.extend(population[i][j][0:cpoint])
                    tmp1.extend(population[i][j + 1][cpoint:len(population[i][j])])
                    # 将tmp2作为暂存器，暂时存放第i个染色体第j+1个取值中的前0到cpoint个基因，
                    # 然后再把第i个染色体第j个取值中的后面的基因，补充到tem2后面
                    tmp2.extend(population[i][j + 1][0:cpoint])
                    tmp2.extend(population[i][j][cpoint:len(population[i][j])])
                    # 将交叉后的染色体取值放入新的种群中
                    population[i][j] = tmp1
                    population[i][j + 1] = tmp2

    ### 1.5 变异操作
    def mutation(self, population):
        '''变异操作
        input:self(object):定义的类参数
              population(list):当前种群
        '''
        pop_len = len(population[0])  # 种群数
        Gene_len = len(population[0][0])  # 基因长度
        for i in range(len(population)):
            for j in range(pop_len):
                if (random.random() < self.pm):
                    mpoint = random.randint(0, Gene_len - 1)  ##基因变异位点
                    ##将第mpoint个基因点随机变异，变为0或者1
                    if (population[i][j][mpoint] == 1):
                        population[i][j][mpoint] = 0
                    else:
                        population[i][j][mpoint] = 1

    ### 1.6 找出当前种群中最好的适应度和对应的参数值
    def best(self, population_decimalism, fitness_value):
        '''找出最好的适应度和对应的参数值
        input:self(object):定义的类参数
              population(list):当前种群
              fitness_value:当前适应度函数值列表
        output:[bestparameters,bestfitness]:最优参数和最优适应度函数值
        '''
        pop_len = len(population_decimalism[0])
        bestparameters = []  ##用于存储当前种群最优适应度函数值对应的参数
        bestfitness = 0.0  ##用于存储当前种群最优适应度函数值

        for i in range(0, pop_len):
            tmp = []
            cs_1 = []
            cs_2 = []
            if (fitness_value[i] > bestfitness):
                bestfitness = fitness_value[i]
                for j in range(len(population_decimalism)):
                    tmp.append(
                        abs(population_decimalism[j][i] / (math.pow(2, self.choromosome_length) - 1)))
                    bestparameters = tmp

                canshu_1 = [8000, 10000]
                canshu_2 = [1, 10]

                cs_1 = tmp[0] * (canshu_1[1] - canshu_1[0]) + canshu_1[0]
                cs_2 = tmp[1] * (canshu_2[1] - canshu_2[0]) + canshu_2[0]
                bestparameters[0] = int(cs_1 // 100)
                bestparameters[1] = abs(round(cs_2 * 0.001, 3))

        return bestparameters, bestfitness

    ### 1.7 画出适应度函数值变化图
    # def plot(self, results):
    #     '''画图
    #     '''
    #     X = []
    #     Y = []
    #     for i in range(self.iter_num):
    #         X.append(i + 1)
    #         Y.append(results[i])
    #     plt.plot(X, Y)
    #     plt.xlabel('Number of iteration', size=15)
    #     plt.ylabel('Value of CV', size=15)
    #     # plt.title('GA_RBF_SVM parameter optimization')
    #     plt.title('GA_RBF_RBDT parameter optimization')
    #     plt.show()

    ### 1.8 主函数
    def main(self):
        results = []  # ！交叉验证结果
        parameters = []  # ！参数结果
        best_fitness = 0.0  # ！初始最好适应度
        best_parameters = []  # ！初始最好参数
        ## 初始化种群
        population = self.species_origin()
        # 迭代参数寻优
        for i in range(self.iter_num):
            # 计算适应函数数值列表
            fitness_value = self.fitness(population)
            # 计算当前种群每个染色体的10进制取值
            population_decimalism = self.translation(population)
            ## 寻找当前种群最好的参数值和最优适应度函数值
            current_parameters, current_fitness = self.best(population_decimalism, fitness_value)
            ## 与之前的最优适应度函数值比较，如果更优秀则替换最优适应度函数值和对应的参数
            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_parameters = current_parameters
            # print('iteration is :', i, ';Best parameters:', best_parameters, ';Best fitness', best_fitness)
            results.append(best_fitness)
            parameters.append(best_parameters)

            ## 种群更新
            ## 选择
            self.selection(population, fitness_value)
            ## 交叉
            self.crossover(population)
            ## 变异
            self.mutation(population)
        results.sort()
        return best_parameters
        # self.plot(results)
        # print('Final parameters are :', parameters[-1])


def diaoyong(population_size, chromosome_num, chromosome_length, max_value, iter_num, pc, pm,
             X_train, X_test, y_train, y_test):
    ga = GA(population_size, chromosome_num, chromosome_length, max_value, iter_num, pc, pm,
            X_train, X_test, y_train, y_test)
    best_parameters = ga.main()
    return best_parameters


# df = pd.read_excel('第8组数据.xlsx')
# X = df.drop(columns=(['输出参数1', '序号', '输出参数2']))
# y = df['输出参数1']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
# population_size = 20
# chromosome_num = 2  # 2
# max_value = 10
# chromosome_length = 20
# iter_num = 100
# pc = 0.6
# pm = 0.01
# res=diaoyong(population_size, chromosome_num, chromosome_length, max_value, iter_num, pc, pm,
#              X_train, X_test, y_train, y_test)
# print(res)