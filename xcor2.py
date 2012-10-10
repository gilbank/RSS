import numpy as np

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
