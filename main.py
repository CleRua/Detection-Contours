
## 1 - Importation des bibliothèques


import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import scipy.signal


## 2 - Chargement de l'image


def load(name):
    img = scipy.ndimage.imread(name)
    img_gray = 0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]
    return(img_gray)


img_test = load("test.gif")

plt.figure()
plt.title("Image de test")
plt.imshow(img_test, cmap="gray")


## 3 - Filtrage

# 3.1 - Filtrage en dimention 1


def diff(x):
    n = len(x)
    D = np.zeros(n)
    D[0] = 0.5 * x[1]
    D[n-1] = -0.5 * x[n-2]
    for i in range(1,n-2):
        D[i] = 0.5 * (x[i+1]-x[i-1])
    return(D)


def applyFilter(h,x): # h est un tuple (h_vecteur, h_indice_0)
    h_vect = h[0]
    h_ind = h[1]
    x_prime = np.concatenate((np.zeros(h_ind),x))
    return(np.convolve(x_prime, h_vect, "same")[h_ind:]) # On recupère la convolution qu'a partir de h_ind pour recevoir un array de la meme taille que x (et pas x_prime). Ceci a pour effet d'éliminer les 0 présents au début de la convolution (information inutile)


def smooth(x):
    h = np.absolute(np.sin(np.linspace(0, np.pi , 81)))
    somme = 0
    for n in range(0,len(h)):
        somme += h[n]
    h = h/somme
    return(applyFilter((h,41), x))


##############

sig1Da = np.zeros((1000));

sig1Da[190:200] = np.sin(np.linspace(-np.pi/2, np.pi/2 , 10))/2 + 1/2;
sig1Da[200:600] = 1;
sig1Da[600:800] = -1;
sig1Da[600:620] = np.sin(np.linspace(np.pi/2, -np.pi/2, 20))
sig1Da[800:830] = np.sin(np.linspace(-np.pi/2, np.pi/2, 30))/2 - 1/2

noise = np.random.randn((1000))

sig1Db = sig1Da + noise/4

###############

### Test fonction diff
diffa = diff(sig1Da)
diffb = diff(sig1Db)
###

### Test fonction applyFilter
h = ([0.5, 0, -0.5], 1)
applya = applyFilter(h, sig1Da)
applyb = applyFilter(h, sig1Db)
###

### Test fonction smooth
smootha = smooth(sig1Da)
smoothb = smooth(sig1Db)
###


### Affichages permettant de tester les fonctions diff, applyFilter et smooth

plt.figure()
plt.title("Affichage de sig1Da, applya et diffa")
plt.plot(sig1Da)
plt.plot(applya)
plt.plot(diffa) # applya et diffa se superposent (fonctionnement attendu)

plt.figure()
plt.title("Affichage de sig1Db, smoothb et sig1Da")
plt.plot(sig1Db)
plt.plot(smoothb) # La fonction smooth a pour effet de lisser le signal bruité sig1Db pour se rapprocher du signal d'origine sig1Da
plt.plot(sig1Da)

plt.figure()
plt.title("Affichage de diffb, diff(smoothb) et diffa")
plt.plot(diffb)
plt.plot(diff(smoothb)) # Le résultat de diff(smoothb) est beaucoup plus exploitable que celui de diffb
plt.plot(diffa)

###


###############


class Filter1D: #Classe filter1D
    def apply(self, signal):
        return
    def ir (self):
        return


class derivator(Filter1D): #Herite de la classe filter1D
    def apply(self, signal):
        return diff(signal)
    def ir(self):
        h_vect = [(0.5) , 0 , (-0.5)]
        return((h_vect, 1))


class filter1D_ir(Filter1D):
    def __init__(self, h) :
        self.h_vect = h[0]
        self.h_ind = h[1]
    def apply(self, signal):
        h = (self.h_vect, self.h_ind)
        return(applyFilter(h, signal))
    def ir(self):
        return((self.h_vect, self.h_ind))


# 3.2 - Filtrage en dimention 2


class raisedcosine:
    def __init__(self, width):
        self.width = width
        k = np.linspace(-self.width, self.width, 2*self.width + 1)
        self.h_vect = 0.5 * (1 + np.cos(np.pi*k/self.width))
    def apply(self, signal):
        return(applyFilter((self.h_vect, self.width), signal))   
    def ir(self):
        return((self.h_vect, self.width))


def applyFilter2D(h, signal):
    h_vect = h[0]
    h_ind = h[1]
    size = np.shape(signal)
    signal_prime = np.zeros((size[0]+h_ind[0], size[1]+h_ind[1]))
    signal_prime[-size[0]:, -size[1]:] = signal
    return(scipy.signal.convolve2d(signal_prime, h_vect, mode="same")[h_ind[0]:, h_ind[1]:])


class filter2D_ir:
    def __init__(self, h):
        self.h = h
    def apply(self, signal):
        return(applyFilter2D(self.h, signal))
    def ir(self):
        return(self.h)


class separable_filter :
    def __init__(self, f1, f2):
        self.f1 = f1
        self.f2 = f2
    def apply (self, signal):
        size = np.shape(signal)
        conv = np.zeros((size[0], size[1]))
        for k in range(size[1]):
            conv[:,k] = self.f1.apply(signal[:,k])
        for k in range(size[0]):
            conv[k,:] = self.f2.apply(conv[k,:])
        return(conv)
    def ir(self):
        h1 = self.f1.ir()
        h2 = self.f2.ir()
        k = np.shape(h1[0])[0]
        l = np.shape(h2[0])[0]
        h = np.zeros((k,l))
        for i in range(k):
            for j in range(l):
                h[i,j] = h1[0][j]*h2[0][i]
        return((h, (h1[1], h2[1])))


# 3.3 - Filtres pour la détection de contour


m = filter1D_ir(([1,2,1],1))
d = filter1D_ir(([1,0,-1],1))

sobel_filter_x = separable_filter(d,m)
sobel_filter_y = separable_filter(m,d)

plt.figure()
plt.title("Filtre de Sobel selon l'axe X")
plt.imshow(sobel_filter_x.apply(img_test), cmap="gray")

plt.figure()
plt.title("Filtre de Sobel selon l'axe Y")
plt.imshow(sobel_filter_y.apply(img_test), cmap="gray")


def sobel(signal):
    m = filter1D_ir(([1,2,1],1))
    d = filter1D_ir(([1,0,-1],1))
    sobel_filter_x = separable_filter(d,m)
    sobel_filter_y = separable_filter(m,d)
    sobel_conv_x = sobel_filter_x.apply(signal)
    sobel_conv_y = sobel_filter_y.apply(signal)
    size = signal.shape
    norme = np.zeros(size)
    angle = np.zeros(size)
    for i in range(size[0]):
        for j in range(size[1]):
            norme[i,j] = np.sqrt(sobel_conv_x[i,j]**2 + sobel_conv_y[i,j]**2)
            angle[i,j] = np.arctan2(sobel_conv_y[i,j], sobel_conv_x[i,j])  
    return((norme, angle))


sobel_test = sobel(img_test)

plt.figure()
plt.title("Norme de la convolution de img_test par le filtre de Sobel")
plt.imshow(sobel_test[0], cmap="gray")

plt.figure()
plt.title("Angle de la convolution de img_test par le filtre de Sobel")
plt.imshow(sobel_test[1], cmap="gray")


## 4 - Méthode de Canny

# 4.1 - Lissage


class gaussian_filter:
    def __init__(self, k, sigma):
        self.k = k
        self.sigma = sigma
        self.h_vect = np.zeros((2*k, 2*k))
        somme = 0
        for n in range(-k,k):
            for m in range(-k,k):
                temp = np.exp(-(n**2 + m**2)/(2 * (sigma**2)))
                somme += temp
                self.h_vect[n+k,m+k] = temp
        self.h_vect = self.h_vect/somme
    def apply(self, signal):
        return(applyFilter2D((self.h_vect, (self.k, self.k)), signal))
    def ir(self):
        return((self.h_vect, (self.k, self.k)))


def smoothing(signal, sigma):
    k = 20 # Valeur de k ?
    gauss_filter = gaussian_filter(k, sigma)
    return(gauss_filter.apply(signal))


plt.figure()
plt.title("Floutage gaussien de img_test pour simga = 5")
plt.imshow(smoothing(img_test, 5), cmap="gray")


# 4.2 - Détection des contours


# Pour discrétiser l'angle du grandient il faut faire des comparaisons pour partager en 8 parties
# ensuite on compare les deux pixels avec le pixel considéré suivant la direction du gradient, s'ils sont plus petits c'est un max local!

def thinning(norme_angle):
    norme = norme_angle[0]
    angle = norme_angle[1]
    size = np.shape(norme)
    norme_thinning = np.copy(norme)
    for i in range (0,size[0]):
        for j in range (0,size[1]):
            #indentation de discrétisation de la direction d'abord la direction
            if((angle[i,j] > 7*np.pi/8 or angle[i,j] < -7*np.pi/8) and i > 0): # angle[i,j] = (-1,0)
                if(norme[i,j] < norme[i-1,j]): # on passe au pixel adjacent pour voir qui est le plus grand
                        norme_thinning[i,j] = 0
            elif((-7*np.pi/8 < angle [i,j] < -5*np.pi/8) and i > 0 and j > 0): # angle[i,j] = (-1,-1)
                if(norme[i,j] < norme[i-1,j-1]):
                        norme_thinning[i,j] = 0
            elif((-5*np.pi/8 < angle [i,j] < -3*np.pi/8) and j > 0): # angle[i,j]= (0,-1)
                if norme[i,j] < norme[i,j-1] :
                        norme_thinning[i,j] = 0
            elif((-3*np.pi/8 < angle [i,j] < -np.pi/8) and i < size[0]-1 and j > 0): # angle[i,j]= (1,-1)
                if(norme[i,j] < norme[i+1,j-1]):
                        norme_thinning[i,j] = 0
            elif((-np.pi/8 < angle [i,j] < np.pi/8) and i < size[0]-1): # angle[i,j]= (1,0)
                if(norme[i,j] < norme[i+1,j]):
                        norme_thinning[i,j] = 0
            elif((np.pi/8 < angle [i,j] < 3*np.pi/8) and i < size[0]-1 and j < size[1]-1): # angle[i,j]= (1,1)
                if(norme[i,j] < norme[i+1,j+1]):
                        norme_thinning[i,j] = 0
            elif((3*np.pi/8 < angle [i,j] < 5*np.pi/8) and j < size[1]-1): # angle[i,j]= (0,1)
                if(norme[i,j] < norme[i,j+1]):
                        norme_thinning[i,j] = 0
            elif((5*np.pi/8 < angle[i,j] < 7*np.pi/8) and i > 0 and j < size[1]-1): # angle[i,j]= (-1,1)
                if(norme[i,j] < norme[i-1,j+1]):
                        norme_thinning[i,j] = 0
    return(norme_thinning)


plt.figure()
plt.title("'Thinning' de la norme de img_test")
plt.imshow(thinning(sobel(img_test)), cmap="gray")


def thresholding(signal, t1, t2): 
    norme_thinning = thinning(sobel(signal))
    size = np.shape(signal)
    bords_forts = np.zeros(size)
    bords_faibles = np.zeros(size)
    for i in range(size[0]):
        for j in range(size[1]):
            if(norme_thinning[i,j] > t2):
                bords_forts[i,j] = 2
            elif(t1 < norme_thinning[i,j] < t2):
                bords_faibles[i,j] = 1
    bords = bords_forts + bords_faibles
    return(bords_forts,bords_faibles,bords)


# Paramètres de floutage et de seuil pour les bords faibles et forts
t1 = 90
t2 = 260
sigma = 0.5

img_threshold = thresholding(smoothing(img_test,sigma), t1, t2)

plt.figure()
plt.title("Bords forts de img_test")
plt.imshow(img_threshold[0], cmap="gray")

plt.figure()
plt.title("Bords faibles de img_test")
plt.imshow(img_threshold[1], cmap="gray")

plt.figure()
plt.title("Bords de img_test")
plt.imshow(img_threshold[2], cmap="gray")


# 4.3 - Hystéresis

         
def hysterisis(bords):
    size = np.shape(bords)
    composantes_connexes = np.zeros(size)
    num_composante = 0
    for i in range(size[0]):
        for j in range(size[1]):
            if(bords[i,j] >= 2 and composantes_connexes[i,j] == 0): # Si on tombe sur un bords fort qui n'appartient à aucune composante connexe connue
                num_composante += 1
                L = [] # Les listes sous python permettent d'implémenter des piles grace aux méthodes L.append(x) et x = L.pop()
                L.append((i,j))
                while(L != []):
                    p = L.pop()
                    composantes_connexes[p[0],p[1]] = num_composante
                    if(p[0] < size[0]-1 and bords[p[0]+1,p[1]] >= 1 and composantes_connexes[p[0]+1,p[1]] != num_composante):
                        L.append((p[0]+1,p[1]))
                    if(p[1] < size[1]-1 and bords[p[0],p[1]+1] >= 1 and composantes_connexes[p[0],p[1]+1] != num_composante):
                        L.append((p[0],p[1]+1))
                    if(p[1] < size[1]-1 and p[0] < size[0]-1 and bords[p[0]+1,p[1]+1] >= 1 and composantes_connexes[p[0]+1,p[1]+1] != num_composante):
                        L.append((p[0]+1,p[1]+1))
                    if(p[0] > 0 and bords[p[0]-1,p[1]] >= 1 and composantes_connexes[p[0]-1,p[1]] != num_composante):
                        L.append((p[0]-1,p[1]))
                    if(p[1] > 0 and bords[p[0],p[1]-1] >= 1 and composantes_connexes[p[0],p[1]-1] != num_composante):
                        L.append((p[0],p[1]-1))
                    if(p[0] > 0 and p[1] > 0 and bords[p[0]-1,p[1]-1] >= 1 and composantes_connexes[p[0]-1,p[1]-1] != num_composante):
                        L.append((p[0]-1,p[1]-1))
                    if(p[0] > 0 and p[1] < size[1]-1 and bords[p[0]-1,p[1]+1] >= 1 and composantes_connexes[p[0]-1,p[1]+1] != num_composante):
                        L.append((p[0]-1,p[1]+1))
                    if(p[0] < size[0]-1 and p[1] > 0 and bords[p[0]+1,p[1]-1] >= 1 and composantes_connexes[p[0]+1,p[1]-1] != num_composante):
                        L.append((p[0]+1,p[1]-1))
    return(composantes_connexes) # Les pixels qui n'appartiennent pas à aucun bord forment la composante connexe numéro 0.


img_hysterisis = hysterisis(img_threshold[2])

plt.figure()
plt.title("Composantes connexes de img_test (numérotées)")
plt.text(np.shape(img_hysterisis)[0]+20, np.shape(img_hysterisis)[1]/2, "Nombre de composantes de l'image : " + str(int(img_hysterisis.max())))
plt.imshow(img_hysterisis)


def enlever_numerotation(composantes_connexes):
    size = np.shape(composantes_connexes)
    for i in range(size[0]):
        for j in range(size[1]):
            if(composantes_connexes[i,j] >= 1):
                composantes_connexes[i,j] = 1
    return(composantes_connexes)


plt.figure()
plt.title("Composantes connexes de img_test (sans numérotation)")
plt.imshow(enlever_numerotation(img_hysterisis))


## 5 - Segmentation


def segmentation(bords_hysterisis):
    size = np.shape(bords_hysterisis)
    composantes_connexes = np.zeros(size)
    num_composante = 0
    for i in range(size[0]):
        for j in range(size[1]):
            if(composantes_connexes[i,j] == 0 and bords_hysterisis[i,j] == 0): # Point qui n'appartient à aucune composante connexe connue et à aucun bord
                num_composante += 1
                L = [] 
                L.append((i,j))
                while(L != []):
                    p = L.pop()
                    composantes_connexes[p[0],p[1]] = num_composante
                    if(p[0] < size[0]-1 and bords_hysterisis[p[0]+1,p[1]] == 0 and composantes_connexes[p[0]+1,p[1]] != num_composante):
                        L.append((p[0]+1,p[1]))
                    if(p[1] < size[1]-1 and bords_hysterisis[p[0],p[1]+1] == 0 and composantes_connexes[p[0],p[1]+1] != num_composante):
                        L.append((p[0],p[1]+1))
                    if(p[0] > 0 and bords_hysterisis[p[0]-1,p[1]] == 0 and composantes_connexes[p[0]-1,p[1]] != num_composante):
                        L.append((p[0]-1,p[1]))
                    if(p[1] > 0 and bords_hysterisis[p[0],p[1]-1] == 0 and composantes_connexes[p[0],p[1]-1] != num_composante):
                        L.append((p[0],p[1]-1))
    return(composantes_connexes)


img_segmentation = segmentation(img_hysterisis)

plt.figure()
plt.title("Segmentation de img_test")
plt.imshow(img_segmentation) 
    

## 6 - Visualisation


def visu():    
    img_smooth = smoothing(img_test, sigma)
    plt.figure()
    plt.title("Lissage de img_test")
    plt.imshow(img_smooth, cmap="gray")
    
    img_composantes = segmentation(hysterisis(thresholding(img_smooth, t1, t2)[2]))
    plt.figure()
    plt.title("Composantes de img_test")
    plt.imshow(img_composantes)
    
    img_Canny = hysterisis(thresholding(img_smooth, t1, t2)[2])
    size = np.shape(img_Canny)
    img_alpha = np.zeros((size[0],size[1],4))
    for i in range(size[0]):
        for j in range(size[1]):
            if(img_Canny[i,j] == 0):
                img_alpha[i,j,0] = 0
                img_alpha[i,j,1] = 0
                img_alpha[i,j,2] = 0
                img_alpha[i,j,3] = 0
            else:
                img_alpha[i,j,0] = 1
                img_alpha[i,j,1] = 255
                img_alpha[i,j,2] = 255
                img_alpha[i,j,3] = 1
    plt.figure()
    plt.title("Superposition de img_test et ses contours")
    plt.imshow(img_test, cmap="gray")
    plt.imshow(img_alpha)
    
    
visu()