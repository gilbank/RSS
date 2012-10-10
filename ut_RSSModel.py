from xcor2 import xcor2


import pylab as pl
import numpy as np

import PySpectrograph
from PySpectrograph.Models import RSSModel
from PySpectrograph.Spectra import Spectrum

import pyfits
import sextractor


def xcor2(inspec,refspec,maxshift,dx=1.,quiet='false'):#       tshift=xcor(tflux,tay0,tay2,maxshift)

    # do crude discrete cross-correlation on each slit (between y0,y1) to find best offset:
    # make running vector from -maxshift,+maxhsift
#    foo1=np.arange(maxshift+1)
#    foo0=-(np.arange(maxshift+1-1)+1) # don't repeat 0.
    #print foo0
    #print foo1
#    tivec=np.append(foo0,foo1)
    #print tivec
#    ivec=np.sort(tivec)
    #print ivec
    ivec=np.arange(-maxshift,maxshift,dx) # doh! - much easier!
        
    #refvec=np.zeros(np.size(inspec))
    #refvec[y0:y1]=1. # Set to top-hat
    #print refvec
    refvec=refspec
    
    xcorval=np.zeros(len(ivec))
    for ii in range(len(ivec)):
        shifrefvec=np.roll(refvec,int(ivec[ii]))
        
#        print np.shape(shifrefvec)
#        print np.shape(inspec)
        xcorval[ii]=np.sum(shifrefvec * inspec)
#        print ii,ivec[ii],xcorval[ii]
        
    #pl.figure()
    #pl.plot(ivec,xcorval)
    pk=np.argmax(xcorval)
    if(quiet=='false'):
        if ((pk==0) | (pk==(len(xcorval)-1))): 
            print
            print 'WARNING: xcor2 peak at edge of window!'
            print 'CHECK RESULTS'
            print
    return ivec[pk],xcorval[pk]







filename='mbxgpP201206140058.fits'
#??data,hdr=pyfits.getdata(filename,extname='sci',header='t')
data=pyfits.getdata(filename)
hdr=pyfits.getheader(filename)
grname=hdr['grating']
camang=hdr['camang']
gratang=hdr['gr-angle']

#create the spectrograph model
rss=RSSModel.RSSModel(grating_name=grname, gratang=gratang, camang=camang, 
                      slit=1.50, xbin=2, ybin=2)


#print out some basic statistics
print 1e7*rss.calc_bluewavelength(), 1e7*rss.calc_centralwavelength(), 1e7*rss.calc_redwavelength()
R=rss.calc_resolution(rss.calc_centralwavelength(), rss.alpha(), -rss.beta())
res=1e7*rss.calc_resolelement(rss.alpha(), -rss.beta())
print R, res

#set up the detector
ycen=rss.detector.get_ypixcenter()
d_arr=rss.detector.make_detector()[ycen,:]
xarr=np.arange(len(d_arr))
w=1e7*rss.get_wavelength(xarr)

#set up the artificial spectrum
sw,sf=pl.loadtxt('Ar.txt', usecols=(0,1), unpack=True)
wrange=[1e7*rss.calc_bluewavelength(), 1e7*rss.calc_redwavelength()]
spec=Spectrum.Spectrum(sw, sf, wrange=wrange, dw=res/10, stype='line', sigma=res)

#interpolate it over the same range as the detector
spec.interp(w)



#plot it
pl.figure()
pl.plot(spec.wavelength, d_arr*((spec.flux)/spec.flux.max()))
pl.plot(spec.wavelength, d_arr*0.1)
yy=np.median(data[1000:1050,3:3173],0)
pl.plot(spec.wavelength,yy/yy.max())


ymod=d_arr*((spec.flux)/spec.flux.max())
ydata=yy/yy.max()

off,rval = xcor2(ydata,ymod,100.)

yyy=np.roll(yy,-int(off))/yy.max()
pl.plot(spec.wavelength,yyy)



pl.show()

if (0):
    sex = sextractor.SExtractor()
    sex.config['CHECKIMAGE_TYPE'] = "SEGMENTATION"
    sex.config['CHECKIMAGE_NAME'] = "seg.fits"
    #sex.config['THRESH_TYPE'] = "ABSOLUTE"
    sex.config['THRESH_TYPE'] = "RELATIVE"
    sex.config['DETECT_THRESH'] = "20"
    sex.config['DETECT_MINAREA'] = "100"
    sex.config['DEBLEND_NTHRESH'] = "1"
    sex.config['BACK_TYPE'] = "MANUAL"
    #sex.config['BACK_VALUE'] = "0.0,0.0"
    
    sex.run(filename)





