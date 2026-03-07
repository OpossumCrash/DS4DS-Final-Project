import numpy as np

def find_center(t):
    time_of_day = t / 24
    time_of_year = t / (24*365)
    max_declination = .4 # Truncated from estimate of earth's solar decline
    lon_center = time_of_day*2*np.pi # Rescale sin to 0-1 then scale to np.pi
    lat_center = np.sin(time_of_year*2*np.pi)*max_declination
    lon_anti = np.pi + lon_center  #2*np.((np.sin(-time_of_day*2*np.pi)+1) / 2)*pi
    return lon_center, lat_center, lon_anti, lat_center

def season_day_forcing(phi, theta, t, h_f0):
    phi_c, theta_c, phi_a, theta_a = find_center(t)
    sigma = np.pi/2
    coefficients = np.cos(phi - phi_c) * np.exp(-(theta-theta_c)**2 / sigma**2)
    forcing = h_f0 * coefficients
    return forcing
        