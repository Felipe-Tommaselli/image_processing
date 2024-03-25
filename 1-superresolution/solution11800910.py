'''
    @Name: Felipe Andrade Garcia Tommaselli
    @Number: 11800910
    @Course: SCC0651
    @Year: 2024.1
    @Title: Assign 1- Image Enhacement
'''

# Importação das bibliotecas necessárias
import numpy as nmp
import imageio.v3 as iiv3


# Definição da função para imprimir arte ASCII
def ascii_art():
    print('/\\'*78)
    print('    __________                                                                       __              .___       .__                                 ')           
    print('\______   _______  ____   ____  ____   ______ __________    _____   ____   _____/  |_ ____     __| _/____   |__| _____ _____    ____   ____   ____   ______')
    print('|     ___\_  __ \/  _ \_/ ____/ __ \ /  ___//  ___\__  \  /     \_/ __ \ /    \   __/  _ \   / __ _/ __ \  |  |/     \\__  \  / ___\_/ __ \ /    \ /  ___/')
    print('|    |    |  | \(  <_> \  \__\  ___/ \___ \ \___ \ / __ \|  Y Y  \  ___/|   |  |  |(  <_> ) / /_/ \  ___/  |  |  Y Y  \/ __ \/ /_/  \  ___/|   |  \\___ \ ')
    print('|____|    |__|   \____/ \___  \___  /____  /____  (____  |__|_|  /\___  |___|  |__| \____/  \____ |\___  > |__|__|_|  (____  \___  / \___  |___|  /____  >')
    print('                            \/    \/     \/     \/     \/      \/     \/     \/                  \/    \/           \/     \/_____/      \/     \/    ')
    print('/\\'*78)

# Definição da função para calcular o histograma de uma imagem
def histo(image, num_levels = 256):
    hist = nmp.zeros(num_levels).astype(int)

    for i in range(num_levels):
        num_pix_val_i = nmp.sum(image == i)
    
        hist[i] = num_pix_val_i

    return hist

# Definição da função para realizar a equalização de histograma em uma única imagem
def sole_cumulative_histo(images, num_levels = 256):
    histo_transf_list = []
    image_eq_list = []

    for image in images:
        hist = histo(image, num_levels)

        histC = nmp.zeros(num_levels).astype(int)
        histC[0] = hist[0]

        for i in range(1, num_levels):
            histC[i] = hist[i] + histC[i - 1]

        hist_transf = nmp.zeros(num_levels).astype(nmp.uint8)

        N, M = image.shape
        image_eq = nmp.zeros([N, M]).astype(nmp.uint8)

        for z in range(num_levels):
            s = ((num_levels - 1) / float(M * N)) * histC[z]
            hist_transf[z] = s

            image_eq[nmp.where(image == z)] = s

        histo_transf_list.append(hist_transf)
        image_eq_list.append(image_eq)

    return image_eq_list, histo_transf_list

# Definição da função para realizar a super-resolução
def hyper_resolution(low_images, high_image):
    h, w = high_image.shape
    low_image_up = nmp.zeros((h, w)) 

    for _ in range(0, h):
        for __ in range(0, w):
            if _ % 2 == 0 and __ % 2 == 0:
                index = 0
            elif _ % 2 == 1 and __ % 2 == 0:
                index = 1
            elif _ % 2 == 0 and __ % 2 == 1:
                index = 2
            elif _ % 2 == 1 and __ % 2 == 1:
                index = 3
            low_image_up[_, __] = low_images[index][_ // 2, __ // 2]
        low_image_up = low_image_up.astype(nmp.uint8)

    output = round(nmp.sqrt(nmp.mean((low_image_up.astype(nmp.int16) - high_image.astype(nmp.int16)) ** 2)), 4)

    return output

# Definição da função para realizar a equalização de histograma conjunta
def joint_cumulative_histo(images, num_levels = 256):
    histo_images = []    
    histo_transf_list = []
    image_eq_list = []

    for image in images:
        hist = histo(image, num_levels)

        histC = nmp.zeros(num_levels).astype(int)
        histC[0] = hist[0]

        for i in range(1, num_levels):
            histC[i] = hist[i] + histC[i - 1]

        histo_images.append(histC)

    histC = nmp.zeros(num_levels).astype(int)

    for i in range(0, 4):
        histC += histo_images[i]

    for image in images:
        hist_transf = nmp.zeros(num_levels).astype(nmp.uint8)

        N, M = image.shape
        image_eq = nmp.zeros([N, M]).astype(nmp.uint8)

        for z in range(num_levels):
            s = ((1/4) * (num_levels - 1) / float(M * N)) * histC[z]
            hist_transf[z] = s

            image_eq[nmp.where(image == z)] = s

        histo_transf_list.append(hist_transf)
        image_eq_list.append(image_eq)

    return image_eq_list, histo_transf_list

# Definição da função para realizar a correção gama em imagens
def gamma_correct(images, gamma_val):
    image_eq_list = []

    for image in images:
        image_eq = (255 * ((image / 255) ** (1 / gamma_val))).astype(nmp.uint8) 
        image_eq_list.append(image_eq)

    return image_eq_list

# Definição da função para selecionar o tipo de pré-processamento a ser aplicado nas imagens
def select_kp(f_option, low_images, gamma_value):
    if f_option == 0:
        out_image = low_images

    elif f_option == 1:
        out_image, _ = sole_cumulative_histo(low_images)

    elif f_option == 2:
        out_image, _ = joint_cumulative_histo(low_images)

    elif f_option == 3:
        out_image = gamma_correct(low_images, gamma_value)
    
    return out_image


# Função principal que executa o código
if __name__ == "__main__":
    # Chamada para imprimir arte ASCII
    #ascii_art()
    # Entrada dos nomes das imagens e outras informações pelo usuário
    image_low_res = input().rstrip()
    image_high_res = input().rstrip()
    f_option = int(input().rstrip())
    gamma_value = float(input().rstrip())

    # Carregamento das imagens de baixa resolução e da imagem de alta resolução
    low_images = list()
    for typ in range(0, 4):
        #image = iiv3.imread('test_cases/' + str(image_low_res) + str(typ) + '.png')
        image = iiv3.imread(str(image_low_res) + str(typ) + '.png')
        low_images.append(image)
    #image_high_res = iiv3.imread('test_cases/' + image_high_res)
    image_high_res = iiv3.imread(image_high_res)

    # Impressão do resultado final
    #print('\nErro final:', end=' ')

    # Seleção do tipo de pré-processamento e aplicação
    out_image = select_kp(f_option, low_images, gamma_value)
    error = hyper_resolution(out_image, image_high_res)

    print(f'{error:.4f}')
