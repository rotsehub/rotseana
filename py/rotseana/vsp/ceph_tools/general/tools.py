import math

def mag2flux( in_mag ):

    out_flux = float((3.636)*((10)**(((-1)*float(in_mag) / (2.5)))))

    return out_flux

def flux2mag( in_flux ):
    
    out_mag = float((-2.5)*math.log10(float(in_flux) / (3.636)))

    return out_mag

def avmag( data ):
    
    fluxs = list()
    for row in data:
        mag = row[1]
        flux = mag2flux(mag)
        fluxs.append(flux)

    avflux = math.fsum(fluxs) / len(fluxs)
    avmag = flux2mag(avflux)

    return avmag

