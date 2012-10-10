import pylab as plt
import pylab as pl
import numpy as np

import PySpectrograph
from PySpectrograph.Models import RSSModel
from PySpectrograph.Spectra import Spectrum

from specutils import *
from RSSutils import *

import pyfits

from numpy import linalg

import scikits.statsmodels.api as sm

import pickle
import time

from scipy import interpolate as intrp
from scipy.signal import cspline1d, cspline1d_eval

from scipy.ndimage import geometric_transform
#from scipy.ndimage.interpolation import map_coordinates

# http://projects.scipy.org/scipy/browser/trunk/scipy/stats/models/robust/scale.py?rev=4460
def unsqueeze(data, axis, oldshape):
    """
        unsqueeze a collapsed array
    
        >>> from numpy import mean
        >>> from numpy.random import standard_normal
        >>> x = standard_normal((3,4,5))
        >>> m = mean(x, axis=1)
        >>> m.shape
        (3, 5)
        >>> m = unsqueeze(m, 1, x.shape)
        >>> m.shape
        (3, 1, 5)
        >>>
    """
    newshape = list(oldshape)
    newshape[axis] = 1
    return data.reshape(newshape)


def MAD(a, c=0.6745, axis=0):
    """
            Median Absolute Deviation along given axis of an array:
    
            median(abs(a - median(a))) / c
    
            """
    
    a = np.asarray(a, np.float64)
    d = np.median(a, axis=axis)
    d = unsqueeze(d, axis, a.shape)
    
    return np.median(np.fabs(a - d) / c, axis=axis)
   
idebug=0

##from polywarp import *

# ==========================================================================
# Modelled on s74spec.py (which is in turn based on doRSS.py
#
# USe Steve's model arc lines instead of a manually-calibrated reference
# 
# ==========================================================================


    
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
    
    


# ==========================================================================


# == pysalt specific mods ==


# -- generate linelist file in same format as _tmparc.lis before
#    Input is just the arc image header to provide relevant parameters

def arclisfromhdr(hdr,slitwidth=1.50,xbin=2,ybin=2,lamp='Ar.txt'):

    # ** some numbers below hardwired for 2x2 binning (~3170 pix) **
    grname=hdr['grating']
    camang=hdr['camang']
    gratang=hdr['gr-angle']

    rss=RSSModel.RSSModel(grating_name=grname, gratang=gratang, camang=camang, 
                      slit=slitwidth, xbin=xbin, ybin=ybin)

    
    # set up the detector
    ycen=rss.detector.get_ypixcenter()
    d_arr=rss.detector.make_detector()[ycen,:]
    xarr=np.arange(len(d_arr))
    w=1e7*rss.get_wavelength(xarr)

    #set up the artificial spectrum
    res=1e7*rss.calc_resolelement(rss.alpha(), -rss.beta())

    sw,sf=pl.loadtxt(lamp, usecols=(0,1), unpack=True)
    wrange=[1e7*rss.calc_bluewavelength(), 1e7*rss.calc_redwavelength()]
    spec=Spectrum.Spectrum(sw, sf, wrange=wrange, dw=res/10, stype='line', sigma=res)
    
    #interpolate it over the same range as the detector
    spec.interp(w)

    if(0):
        #plot it
        pl.figure()
        pl.plot(spec.wavelength, d_arr*((spec.flux)/spec.flux.max()))
        pl.plot(spec.wavelength, d_arr*0.1)
        #yy=np.median(data[1000:1050,3:3173],0)
        #pl.plot(spec.wavelength,yy/yy.max())
        
        
        ymod=d_arr*((spec.flux)/spec.flux.max())
        ydata=yy/yy.max()
        
        off,rval = xcor2(ydata,ymod,100.)
        
        yyy=np.roll(yy,-int(off))/yy.max()
        pl.plot(spec.wavelength,yyy)
        pl.show()
        stop()
        
    # We need to return 
    # - a matched list of wavelength(of each pixel),flux for the arc
    # - a list of the arc lines
    modarclam=spec.wavelength
    modarcspec=d_arr*((spec.flux)/spec.flux.max())
    
    # extract pixel positions for lines of wavelength sw:
    xpix=np.arange(np.size(modarclam))
    ixp = np.interp(sw, modarclam, xpix, left=0, right=0)
    
    ok=np.reshape( ((ixp>0.)&(ixp<3170)).nonzero(),-1 )
    
    np.savetxt('_tmparc.lis',np.transpose((ixp[ok],sw[ok],sw[ok])))
    
    return modarclam,modarcspec


def dummymapping((xout,yout),okx,oky):
    xin,yin=applywarp(xout,yout,okx,oky)
    return (xin,yin)

def DUMBdummymapping((xout,yout)):
    # dummy function to use polwarp coeffs and feed to geometric_transform
    
    """
    mapping must be a callable object that accepts a tuple of length equal to the 
    output array rank and returns the corresponding input coordinates as a 
    tuple of length equal to the input array rank. 
    The output shape and output type can optionally be provided. 
    If not given they are equal to the input shape and type.
    """
    
    # read coeffs from pickle file
    pklfile = open('mapping.pkl', 'rb')
    okx=pickle.load(pklfile)
    oky=pickle.load(pklfile)
    pklfile.close()
    xin,yin=applywarp(xout,yout,okx,oky)
    
    
    #return (xin.tolist(),yin.tolist())
    return (xin,yin)

    
    
#******************************************************************************************
#stop()
    
idebug=0

filename='mbxgpP201206140058.fits'
#??data,hdr=pyfits.getdata(filename,extname='sci',header='t')
data=pyfits.getdata(filename)
hdr=pyfits.getheader(filename)

# central[ish] row of image
yy=np.median(data[1000:1050,3:3173],0)
#yy=np.median(data[1500:1550,3:3173],0)
#arcspec=np.median(data[1000:1050,3:3173],0)
arcspec=yy/yy.max()

reflam,refspec = arclisfromhdr(hdr)

#lamcal = wlc_arc(arcspec,reflam,refspec)
# plt.show()

# -- New method. Fit WLC at one row only (central):

yc=int(data.shape[0]/2.)
y0=yc-20
y1=yc+20

yy=np.median(data[y0:y1,3:3173],0)
arcspec=yy/yy.max()
tlamcal = wlc_arc(arcspec,reflam,refspec,lenfrac=1.0)

arclines,ht=np.loadtxt('Ar.txt',unpack=True,skiprows=1)
ok=np.where((ht >10000.))[0]
arclines=arclines[ok]
# map arclines to pixel coords (approx):
xx=np.arange(len(tlamcal))

linex=np.interp(arclines,tlamcal,xx)
ok=np.where((linex>50)&(linex<linex.max()-50))
linex=linex[ok]
ybord=100
print 'tracing lines in spatial direction...'
kx,ky,xo,yo=tracelines(data,linex,yref=yc,ystep=50,ybord=ybord,lineFWHM=7.)#,idebug=True)
print 'done.'
#def transformByRow(image,kx,ky):

#ybord=800
# -- transform row-by-row:
print 'calculating/applying rectification...'
#plt.figure()
ny,nx=np.shape(data)
xx=np.arange(nx)
odata=np.zeros((ny,nx))
for i in np.arange(ny):
    if ((i>ybord)&(i<ny-ybord)):
        ff=data[i,:]
        yy=np.repeat(i,nx)
        xf,yf=applywarp(xx,yy,kx,ky)
##        plt.plot(xf,ff,'k,')
#        plt.plot(np.round(xf),ff,'r,') # c.f. non-sub-pixel method
    # make output rectified image:
        odata[i,:]=np.interp(xx,xf,ff)

##plt.show()
print 'done.'




# ==== Attempt to flatfield ====

# slit fn (trace from spectral average of arc lines in spatial direction):
totslitfn=np.median(odata,axis=1)

# **** maybe smooth with spline?? ****
# Make 2D:
nnn=np.repeat(np.reshape(totslitfn,(-1,1)),(3176),axis=1) #!
#imshow(odata/nnn)
nnnn=nnn/np.median(nnn)
##plt.plot((odata/nnnn).T)

##plt.show()
# *** need to test how well this slit fn calculated from arcs will work on science data...

plt.figure()
# apply slit fn to 2d image and plot (Kelson-)transformed lines...:
oodata=np.zeros((ny,nx))
ndata=data/nnnn # should slit fn. be applied to transformed or untransformed data?
oxf=[]
off=[]
for i in np.arange(ny):
    if ((i>ybord)&(i<ny-ybord)):
        ff=ndata[i,:]
        yy=np.repeat(i,nx)
        xf,yf=applywarp(xx,yy,kx,ky)
#        plt.plot(xf,ff,'k,')
        oxf=np.append(oxf,xf)
        off=np.append(off,ff)
#        plt.plot(np.round(xf),ff,'r,') # c.f. non-sub-pixel method
    # make output rectified image:
        oodata[i,:]=np.interp(xx,xf,ff)

plt.show()

# -- calculate medians and stddevs in bins of nearest pixel:

if (0):
    m=np.median(oodata,axis=0)
    s=MAD(oodata,axis=0)
    
    bins=np.arange(np.shape(oodata)[1])
    inds=np.digitize(oxf,bins)-1
    #iok=np.where((inds >0))[0]
    #inds=inds[iok]
    f0=m-3.*s
    f1=m+3.*s
    #for i in np.arange()
    good=np.where((off>f0[inds]) & (off<=f1[inds]))[0]
    bad=np.where((off<f0[inds]) | (off>f1[inds]))[0]
    plt.plot(oxf[bad],off[bad],'rx')
    plt.plot(oxf[good],off[good],'k,')
    
    plt.plot(m,'b-')
    plt.plot(m+s,'r-')
    plt.plot(m-s,'g-')
    plt.plot(m+3.*s,'r:')
    plt.plot(m-3.*s,'g:')
    plt.show()

# do this in a cleverer way with adaptive binning:
ss=np.argsort(oxf)#[::-1]
soxf=oxf[ss]
soff=off[ss]

#stepsize=100
#sfs=[]
#sfm=[]

# do this a cleverer way with arrays:

#newcollen=200
newcollen=1000
nl=int(float(len(soxf))/float(newcollen))
tl=nl*newcollen

#rebin_xf=np.reshape(soxf[0:tl],(newcollen,nl))
#rebin_ff=np.reshape(soff[0:tl],(newcollen,nl))
rebin_xf=np.reshape(soxf[0:tl],(nl,newcollen))
rebin_ff=np.reshape(soff[0:tl],(nl,newcollen))

m=np.median(rebin_ff,axis=1)
s=MAD(rebin_ff,axis=1)

nsig=3.0
nsig=5.0
tf0=m-nsig*s
tf1=m+nsig*s

# extend 1D array into 2nd dimension for comparison with expected flux range. Complicated!
f0=np.repeat(np.reshape(tf0,(-1,1)),(newcollen),axis=1)
f1=np.repeat(np.reshape(tf1,(-1,1)),(newcollen),axis=1)

#*** i'm not sure if i've done sthg wrong. using end of array instead of centre, for example, 
#    as rejected pixels seem to be systematically on redward side of data...

# i think this is *prob* alright, and it may be a problem with the mapping/flatfielding...
# maybe try a shorter range in the spatial direction and see if better...
good=np.where( (rebin_ff>f0) & (rebin_ff<=f1))#[0]
bad=np.where( (rebin_ff<f0) | (rebin_ff>f1))#[0]
plt.plot(rebin_xf[good],rebin_ff[good],'k,')
plt.plot(rebin_xf[bad],rebin_ff[bad],'rx')

plt.plot(rebin_xf.T[0],m,'b-')

plt.show()


# maybe just try fitting bspline to median?

#x=np.reshape(rebin_xf,-1)
#y=np.reshape(rebin_ff,-1)

x=rebin_xf.T[0]
y=m

#x=tx[ok]
#y=ty[ok]
err=1./np.sqrt(y)
#w=1./err
w=np.ones(len(x))#/1000. # decent estimate of error is crucial here. would be best to cleverly determine extent of useful skylines before getting
# to this stage and having lower estimate of error.
tck=intrp.splrep(x,y,w=w)#,k=4)#w=w)#,s=5)#,xb=1145,xe=1155)

xx=np.arange(0,np.shape(oodata)[1],0.1)
yf=intrp.splev(xx,tck,ext=1)
plt.plot(xx,yf,'mo-')
plt.show()

#**** checked up to here

print 'calculating inverse rectification transformation...'
gridsz=20
# grid in 20 pixel steps (Should be fine for longslit):
gx,gy=np.meshgrid(np.arange(0,np.shape(data)[0],gridsz),np.arange(0,np.shape(data)[1],gridz))
gx=np.reshape(gx,-1)
gy=np.reshape(gy,-1)
gxf,gyf=applywarp(gx,gy,kx,ky)
ikx,iky=polywarp(gx,gy,gxf,gyf,3)
# test:
ny,nx=np.shape(data)
xx=np.arange(nx)
tdata=np.zeros((ny,nx))
for i in np.arange(ny):
    if ((i>ybord)&(i<ny-ybord)):
        ff=oodata[i,:]
        yy=np.repeat(i,nx)
        ixf,iyf=applywarp(xx,yy,ikx,iky)
##        plt.plot(xf,ff,'k,')
#        plt.plot(np.round(xf),ff,'r,') # c.f. non-sub-pixel method
    # make output rectified image:
        iff=intrp.splev(ixf,tck,ext=1)
        tdata[i,:]=iff

plt.figure()
plt.imshow(tdata)
plt.show()


stop()

#for i in np.arange(stepsize,len(soxf),stepsize):
#    tfm=np.median(soff[i-stepsize:i])
#    tfs=MAD(soff[i-stepsize:i])
##    tsxp=np.median(soxp[i-stepsize:i])
#    sfm=np.append(sfm,np.repeat(tfm,stepsize))
#    sfs=np.append(sfs,np.repeat(tfs,stepsize))
#
#sf0=sfm-3.*sfs
#sf1=sfm+3.*sfs

#good=np.where( (soff>sf0) & (soff<=sf1) )[0]
#bad=np.where( (soff<sf0) | (soff>sf1) )[0]

plt.plot(soxf[bad],soff[bad],'rx')
plt.plot(soxf[good],soff[good],'k,')
plt.show()

stop()
 




























step=50
tvec=np.arange(100,1900,step)
lams=np.zeros((np.size(tvec),3170))
ii=-1


for y0 in tvec:
    ii=ii+1
    y1=y0+step
    yy=np.median(data[y0:y1,3:3173],0)
    arcspec=yy/yy.max()
    tlamcal = wlc_arc(arcspec,reflam,refspec,lenfrac=1.0)
    #print np.shape(tlamcal)
    #print np.shape(lams)
    lams[ii]=tlamcal
##    plt.close()

# -- use this mapping to trace arc lines:
ycoo=tvec+float(step/2.)
# -- read in list of arc lines:
lines,strength=np.loadtxt('Ar.txt',unpack=True)
# interpolate each row to find the pixel which corresponds to the arc line

# crop line list to those well on detector (within 1000A of edges):
lines=lines[((lines>lams.min()+100.)&(lines<lams.max()-100.))]

xpix=np.arange(3170)

xp=np.zeros((lams.shape[0],len(lines)))

for ii in range(lams.shape[0]):
    fitarclam=lams[ii]
    ixp = np.interp(lines, fitarclam, xpix, left=0, right=0)
    xp[ii]=ixp


#plt.plot(xp)

# Let's set some limits on y fitting region. 
# -WLC has gone a bit screwy at bottom and top of detector!

ymin=500 # pix
ymax=1500 # pix 

# now loop over a given arc line and fit as fn. of pixel posn.

xo=[]
yo=[]
xi=[]
yi=[]

for ii in range(lines.shape[0]):
    use=np.reshape(((ycoo>ymin)&(ycoo<ymax)).nonzero(),-1)
    #plt.plot(ycoo,xp[:,ii])
    plt.plot(ycoo[use],xp[use,ii])
    # fit cubic to each line:
    pcoeffs=robust_poly_fit(ycoo[use],xp[use,ii],order=3)
    #xx=np.arange(ymin,ymax)
    xx=np.arange(100,1900)
    plt.plot(xx,np.polyval(pcoeffs,xx), 'k:')
    
    #-- the mapping we want is from xp,yp to lambda,yp(currently same, but may want to rectify s-distortion later) 
    xo=np.append(xo,np.reshape(xp[use,ii],-1))
    yo=np.append(yo,ycoo[use])
    tlines=np.repeat(lines[ii],np.size(use))
    xi=np.append(xi,tlines)
    yi=np.append(yi,ycoo[use])
    
plt.show()

# Need to warp these solutions to produce a single mapping for feeding to Kelson-style
# sky line fitter

#****
# should we just do this a dumb way by choosing some nominal reference line (centre of curvature?)
# and interpolating each row of data to agree (based on model cubic fit) ?

# OR do I want a version of IDL polywarp to map x,y to lambda, spatial ???
#****
plt.plot(xo,yo,'k,')
plt.plot(xi,yi,'r,')
kx,ky=polywarp(xi,yi,xo,yo,3)

xf,yf=applywarp(xo,yo,kx,ky)
plt.plot(xf,yf,'rD',mec='r',mfc='none')

# xf is lambda, yf is spatial

##-- for geomap/geotran:
#np.savetxt('geomap.in',np.transpose((xo,yo,(xf-5200),yf)))    
# should do this as a regular grid, really. Otherwise, images is truncated to extent of line list
x=np.linspace(0,data.shape[1],100)
y=np.linspace(0,data.shape[0],100)
xy=np.meshgrid(x,y)
gxo,gyo = np.reshape(xy[0],-1),np.reshape(xy[1],-1)
gxf,gyf=applywarp(gxo,gyo,kx,ky)
# does geotran work in logical or physical pixels???:
np.savetxt('geomap.in',np.transpose(((gxf-5000),gyf,gxo,gyo)))   # geomap wants cols in non-obvious order! 
#np.savetxt('geomap.in',np.transpose(((gxf-5200),gyf,2.*gxo,2.*gyo)))   # geomap wants cols in non-obvious order! 

##--


if(1):
    # Now we want to resample this back to a linear frame. Or a frame with the same wavelength
    # solution as some reference spec.
    
    # let's switch to more useful notation
    
    lam=xf
    yspat=yf
    xp=xo
    yp=yo
    
    xout=lam-5200.
    
    # we need to map between output coords and input:
    ##okx,oky=polywarp(xp,yp,xout,yspat,3)
    okx,oky=polywarp(yp,xp,yspat,xout,3)
    # write okx,oky to pickle file for dummymapping to read
    #okx,oky=polywarp(xout,yspat,xp,yp,3)
    
    
    output = open('mapping.pkl', 'wb')
    pickle.dump(okx,output)
    pickle.dump(oky,output)
    output.close()
    
    # does geometric_transform actually require the inverse transform?
    #NO ookx,ooky=polywarp(yspat,xout,yp,xp,3)
    
#    sdata=data[1000:1150,1000:1150]
    print 'transforming...'
    t0=time.time()
    wdata=geometric_transform(data,dummymapping, extra_arguments=(okx,oky),\
                              output_shape=(2000,1000),order=3)
    t1=time.time()
    print "%.1f seconds\n"%(t1-t0)

    stop()

# -- science frame:
if (0):
    filename1='mbxgpP201206140041.fits'
    data1=pyfits.getdata(filename1)
    filename2='mbxgpP201206140042.fits'
    data2=pyfits.getdata(filename2)
    filename3='mbxgpP201206140043.fits'
    data3=pyfits.getdata(filename3)
    
    tdata=np.array([data1,data2,data3])
    data=np.median(tdata,axis=0)



# dumb, one-line-at-a-time method:
xlam=np.zeros(np.size(data))
flux=np.zeros(np.size(data))
ii=0
n=data.shape[1]
rdata=np.zeros(data.shape)

# generate linear WCS soln:
#crval1=np.min(xf)
#cdelt1=

for i in np.arange(data.shape[0]):
    tflux=data[i,:]
    flux[ii:ii+n]=tflux
    #txlam=np.arange(data.shape[1])
    txo=np.arange(data.shape[1])
    tyo=np.repeat(i,data.shape[1]) # assumes no y-fitting, constant y pix val
    txf,tyf=applywarp(txo,tyo,kx,ky)
    txlam=txf

    xlam[ii:ii+n]=txlam
 
    
    # only use useful section of image
    #if ( (i<900)|(i>1200) ): 
    # exclude bright target!:
    if ( (i>900)&(i<1200) ): 
        flux[ii:ii+n]=0
        xlam[ii:ii+n]=0
    if ( (i>1250)&(i<1350) ): 
        flux[ii:ii+n]=0
        xlam[ii:ii+n]=0
    if ( (i>1900)|(i<200) ): 
        flux[ii:ii+n]=0
        xlam[ii:ii+n]=0
    
 
#    need to set up spec. WCS before doing this!:
#    rtflux = np.interp(txlam, np.arange(data.shape[1]), tflux, left=0, right=0)
#    rdata[i,:]=rtflux
    
    
    #flux=np.append(flux,tflux)
    #xlam=np.append(xlam,txlam)
    ii=ii+n
    # much faster than append method!
    
# plt.plot(xlam,flux,'k,')
# works! - lines much sharper in rectified coords







