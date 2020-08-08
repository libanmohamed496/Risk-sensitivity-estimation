# Risk sensitivity estimation

This repository contains the output produced by a 3-week research project hosted by UMN's Institue for Mathematics and its Applications with industry partners at Securian Financial. The project involved a two-tiered process for estimating risk sensitivities, first by producing Monte Carlo estimates for given market conditions and then by interpolating between those estimates. All work included in this repository is my own ([Liban](https://www.linkedin.com/in/libanmohamed496/)'s), except for five_pt_reg.py, which is due to [Radmir Sultamuratov](https://www.linkedin.com/in/radmir-sultamuratov/).

Where to start: the following is quite **tl;dr**, so feel free to jump in by looking at these two notebooks. For results and pictures, just look at the second:

    gmab_model writeup.ipynb
    Sr regression writeup.ipynb

What's included:

    gmab_model writeup.ipynb, gmab_model.py, gmab_fast.py 

gmab_model writeup.ipynb explains the use of the stochastic model that was developed in the context of this project. gmab_model.py contains this stochastic model. gmab_fast.py is a version that doesn't carry stock price information (didn't end up being much faster, but included for dependency reasons).

    Notes on the samples.ipynb , GMAB_ratchet_samples.zip, GMAB_samples.zip, sample10.zip, and sample11.zip 

The .zip folders contain samples that were generated for the sub-team working on regression in Sr-space. I've omitted the ones I gave to the individuals working on regression in contract-space and the NN attempt. The ipynb contains a brief description of the differences between the samples.

    modelling_workbook.py , trading_grid.py, cubic_spline.py, five_pt_reg.py, and gaussian_process_reg.py 

modelling_workbook.py (sorry for un-American orthography) is a file I mocked up to get the regression team started. The others are loosely based off of that one, where the regression techniques have been made into functions that take file paths pointing to sample data and a target number of MC runs; the functions return a tuple of numpy arrays that contain the absolute error in delta and in rho associated to each point in a grid in Sr-space

This part is neither well-organized nor well-documented. This is largely because the regression sub-team was working in parallel and had mixed levels of Python fluency. For example, each regression technique should ideally be a class, but I felt that it wasn't worth the time to get the other members of the regression team up to speed on OOP. Please feel free to ask if you have any questions about this stuff.

    Sr regression writeup.ipynb, results_workbook.py, and Sr_reg_test_2.npy

Sr regression writeup.ipynb details the results of the regression sub-team. Sr_reg_test_2.npy is a numpy array that contains the results of the simulations used to compare the different techniques; this numpy array was generated in results_workbook.py.

A note about directory structure: most scripts assume that gmab_model.py is in the working directory. Some of the older scripts may still have sample file paths as something like 'sample6/sample6_delta_.csv' instead of 'GMAB_samples/sample6_delta_.csv'. It should be pretty easy to figure out the correct file path - samples 2-5 are in GMAB_ratchet_samples; samples 6-9 are in GMAB_samples.
