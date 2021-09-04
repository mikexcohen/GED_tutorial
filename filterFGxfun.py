import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.signal


def filterFGx(data,srate,f,fwhm,showplot=False):
    '''
    :: filterFGx   Narrow-band filter via frequency-domain Gaussian
    filtdat,empVals]= filterFGx(data,srate,f,fwhm,showplot=0)


      INPUTS
         data : 1 X time or chans X time
        srate : sampling rate in Hz
            f : peak frequency of filter
         fhwm : standard deviation of filter, 
                defined as full-width at half-maximum in Hz
     showplot : set to true to show the frequency-domain filter shape (default=false)

      OUTPUTS
      filtdat : filtered data
      empVals : the empirical frequency and FWHM (in Hz and in ms)

    Empirical frequency and FWHM depend on the sampling rate and the
     number of time points, and may thus be slightly different from
     the requested values.

     mikexcohen@gmail.com
    '''

    ## compute filter

    # frequencies
    hz = np.linspace(0,srate,data.shape[1])

    # create Gaussian
    s  = fwhm*(2*np.pi-1)/(4*np.pi) # normalized width
    x  = hz-f                       # shifted frequencies
    fx = np.exp(-.5*(x/s)**2)       # gaussian
    fx = fx/np.max(fx)              # gain-normalized

    # apply the filter
    filtdat = np.zeros( np.shape(data) )
    for ci in range(filtdat.shape[0]):
        filtdat[ci,:] = 2*np.real( np.fft.ifft( np.fft.fft(data[ci,:])*fx ) )



    ## compute empirical frequency and standard deviation

    empVals = [0,0,0]

    idx = np.argmin(np.abs(hz-f))
    empVals[0] = hz[idx]

    # find values closest to .5 after MINUS before the peak
    empVals[1] = hz[idx-1+np.argmin(np.abs(fx[idx:]-.5))] - hz[np.argmin(np.abs(fx[:idx]-.5))]

    # also temporal FWHM
    tmp  = np.abs(scipy.signal.hilbert(np.real(np.fft.fftshift(np.fft.ifft(fx)))))
    tmp  = tmp / np.max(tmp)
    tx   = np.arange(0,data.shape[1])/srate
    idxt = np.argmax(tmp)

    empVals[2] = (tx[idxt-1+np.argmin(np.abs(tmp[idxt:]-.5))] - tx[np.argmin(np.abs(tmp[0:idxt]-.5))])*1000



    ## inspect the Gaussian (turned off by default)

    # showplot=True

    if showplot:
        plt.subplot(211)
        plt.plot(hz,fx,'o-')
        xx = [ hz[np.argmin(np.abs(fx[:idx]-.5))], hz[idx-1+np.argmin(np.abs(fx[idx:]-.5))] ]
        yy = [ fx[np.argmin(np.abs(fx[:idx]-.5))], fx[idx-1+np.argmin(np.abs(fx[idx:]-.5))] ]
        plt.plot(xx,yy,'k--')
        plt.xlim([np.max(f-10,0),f+10])

        plt.title('Requested: %g, %g Hz; Empirical: %.2f, %.2f Hz' %(f,fwhm,empVals[0],empVals[1]) )
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude gain')

        plt.subplot(212)
        tmp1 = np.real(np.fft.fftshift(np.fft.ifft(fx)))
        tmp1 = tmp1 / np.max(tmp1)
        tmp2 = np.abs(scipy.signal.hilbert(tmp1))
        plt.plot(tx-np.mean(tx),tmp1, tx-np.mean(tx),tmp2)
        plt.xlim([-empVals[2]*2/1000,empVals[2]*2/1000])
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude gain')
    plt.show()
    # 
    
    ## outputs
    return filtdat,empVals