# python implementation of IDL's polywarp

import numpy as np
import pylab as plt

#from numpy.polynomial.polynomial import polyvander
#from numpy.polynomial.polynomial import polyvander2d

"""
;    Xi and Yi are expressed as polynomials of Xo, Yo:
;        Xi = Kx[i,j] * Xo^j * Yo^i   Summed for i,j = 0 to degree.
;    And
;        Yi = Ky[i,j] * Xo^j * Yo^i.
;
;    This coordinate transformation may be then used to
;    map from Xo, Yo coordinates into Xi, Yi coordinates.
"""

# NOTE: USE:
# xf,yf=applywarp(xo,yo,kx,ky)

#** This hasn't been exhaustively tested, but i think it's ok **
def applywarp(xo,yo,kx,ky):
    np1=np.shape(kx)[0]
    n=np1#-1
    m=np.size(xo)
    xi=np.zeros(m)
    yi=np.zeros(m)
    for i in range(n):
        for j in range(n):
            xi=xi+kx[j,i]*(xo**j)*(yo**i)
            #print xi
            yi=yi+ky[j,i]*xo**j*yo**i
    # this python indexing is really confusing!
    # Seems to work now, though!
    # xi and yi are reversed
    
    return xi,yi
#    return yi,xi

#def polyvander2d(x,y,deg):
#    # from future version of numpy
#    ideg = [int(d) for d in deg]
#    is_valid = [id == d and id >= 0 for id, d in zip(ideg, deg)]
#    if is_valid != [1, 1]:
#        raise ValueError("degrees must be non-negative integers")
#    degx, degy = ideg
#    x, y = np.array((x, y), copy=0) + 0.0
#
#    vx = polyvander(x, degx)
#    vy = polyvander(y, degy)
#    v = vx[..., None]*vy[..., None, :]
#    # einsum bug
#    #v = np.einsum("...i,...j->...ij", vx, vy)
#    return v.reshape(v.shape[:-2] + (-1,))



def polywarp(xi,yi,xo,yo, degree):
#if (1):
#    xi=np.arange(10)+5
#    yi=np.arange(10)
#    xo=np.arange(10)#+5.
#    yo=np.arange(10)
#    
#    xi = np.array([24, 35, 102, 92])
#    yi = np.array([81, 24, 25, 92])
#    xo = np.array([61, 62, 143, 133])
#    yo = np.array([89, 34, 38, 105])
#     
#    degree=1
    
    
    m = np.size(xi) # no. of points
# ** error checking. check no.s of elts.
    if (np.size(xo) <> m):
        print 'no. of elements must be same in xo,yo,xi,yi'
        stop()
    
# 
    n = degree        #;use halls notation
    n2=(n+1)**2
    #** if n2 gt m then message, '# of points must be ge (degree+1)^2.'
    if (n2>m):
        print # of points must be >= (degree+1)^2.'
        stop()
        

    x = np.transpose(np.array([np.transpose(xi[:]),np.transpose(yi[:])]))
    u = np.transpose(np.array([np.transpose(xo[:]),np.transpose(yo[:])]))

    ut=np.zeros((m,n2)) #;transpose of U
    u2i = np.zeros(n+1)    #;[1,u2i,u2i^2,...]
    
    for i in np.arange(0,m):
        u2i[0]=1 # ;init u2i
        zz = u[i,1]
        for j in np.arange(1,n+1): u2i[j]=u2i[j-1]*zz
        ut[i,0:n+1]=u2i # ;evaluate 0 th power separately
        for j in np.arange(1,n+1): ut[i,j*(n+1):j*(n+1)+n+1]=u2i*u[i,0]**j # ;fill ut=u0i^j * U2i
#    #
    # above is correct, by c.f. IDL

#    ut=polyvander2d(x,u,(n,n))
#    mut=np.matrix(ut)
    
    uu = np.dot(np.transpose(ut),ut) # ;big u
 #   uu=mut*mut.transpose()
    
    kk = np.linalg.inv(uu) #**** need to add error checking
    okk=kk
    
    kk = np.dot(ut,kk)
    #fkk = np.matrix(kk)*np.matrix(ut)
    #kk=0
    #kk=fkk
    #fkk=0
    
    #kx = np.zeros(n+1,n+1) + (np.multiply(kk,np.transpose(x[:,0])))
    #ky = np.zer
    nkk=np.matrix(np.reshape(kk,-1))
    xT=np.matrix(x).transpose()
    #kx = np.zeros(n+1,n+1) + (xT[0,:]*kk)
    #ky = np.zeros(n+1,n+1) + (xT[1,:]*kk)
    
    
    kx=np.reshape( np.matrix(kk.T)*np.matrix(xT[0,:].T) ,(n+1,n+1))
    ky=np.reshape( np.matrix(kk.T)*np.matrix(xT[1,:].T), (n+1,n+1))
    
    return kx.A,ky.A # turn back into arrays
    
    
def testpw():
    xi = np.array([24, 35, 102, 92])
    yi = np.array([81, 24, 25, 92])
    xo = np.array([61, 62, 143, 133])
    yo = np.array([89, 34, 38, 105])
    degree=1
    
    kx,ky=polywarp(xi,yi,xo,yo,degree)
    
    plt.plot(xo,yo,'ob')
    plt.plot(xi,yi,'ok')    
    
    xf,yf=applywarp(xo,yo,ky,kx)
    plt.plot(xf,yf,'Dr',mfc='none',mec='r')    
    
    x=np.arange(0,200,20)
    y=np.arange(0,200,20)
    xy=np.meshgrid(x,y)
    
    #txo=np.arange(100)
    #tyo=np.arange(100)
    # need to define points on a grid:
    txo,tyo = np.reshape(xy[0],-1),np.reshape(xy[1],-1)
    
    txf,tyf=applywarp(txo,tyo,ky,kx)
    plt.plot(txf,tyf,'Dg',mfc='none',mec='g')    
    
    tkx,tky=polywarp(txf,tyf,txo,tyo,2)
    oxf,oyf=applywarp(txo,tyo,tky,tkx)
    plt.plot(oxf,oyf,'k,')    
    
    dx2=np.sum(( oxf-txf )**2)
    dy2=np.sum(( oyf-tyf )**2)
    
    print "residuals: dx: %.2f, dy: %.2f"%(np.sqrt(dx2),np.sqrt(dy2))
    
    