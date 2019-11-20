from math import pi

def calB(data):
    u0 = 4*pi*(10**(-7))
    N = 500
    R = 0.1
    I = 100
    result = []
    for x in data:
        B =(N*I)*u0*(R**2)/(2*(R**2+x**2)**(3/2))
        result.append(B)
    return result

def percentError(p_data, t_data):
    result = []
    for i in range(len(p_data)):
        result.append(abs(p_data[i]-t_data[i])/abs(p_data[i])*100)
    return result

def calHB(data):
    u0 = 4 * pi * (10 ** (-7))
    N = 500
    R = 0.1
    I = 100
    d = 0.1
    result = []
    for z in data:
        result.append((1/2)*u0*N*I*(R**2)*((R**2 + (d/2 + z)**(2))**(-3/2) + (R**2 + (d/2 - z)**(2))**(-3/2)))
    return result

if __name__ == '__main__':
    # t_data = [0.086, 0.091, 0.115, 0.131, 0.151, 0.168, 0.187, 0.200, 0.213, 0.221, 0.219, 0.213, 0.201, 0.183, 0.166, 0.146, 0.130, 0.112, 0.095]
    data = [i/100 for i in range(-9, 10)]
    print(calHB(data))
    # print(percentError(calB(data), t_data))
