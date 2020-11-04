#ifndef  IC_2NDORDER_HEADER
#define  IC_2NDORDER_HEADER

void displace_pcls_ic_2ndorder(double coeff, double lat_resolution, part_simple * part, double * ref_dist, part_simple_info partInfo, Field<Real> ** fields, Site * sites, int nfield, double * params, double * outputs, int noutputs)
{
	int i;
	Real gradxi[3] = {0, 0, 0};
	
	if (nfield > 1 && (*part).ID % 8 == 0)
		i = 1;
	else
		i = 0;
	
	gradxi[0] = (1.-ref_dist[1]) * (1.-ref_dist[2]) * ((*fields[i])(sites[i]+0) - (*fields[i])(sites[i]));
	gradxi[1] = (1.-ref_dist[0]) * (1.-ref_dist[2]) * ((*fields[i])(sites[i]+1) - (*fields[i])(sites[i]));
	gradxi[2] = (1.-ref_dist[0]) * (1.-ref_dist[1]) * ((*fields[i])(sites[i]+2) - (*fields[i])(sites[i]));
	gradxi[0] += ref_dist[1] * (1.-ref_dist[2]) * ((*fields[i])(sites[i]+1+0) - (*fields[i])(sites[i]+1));
	gradxi[1] += ref_dist[0] * (1.-ref_dist[2]) * ((*fields[i])(sites[i]+1+0) - (*fields[i])(sites[i]+0));
	gradxi[2] += ref_dist[0] * (1.-ref_dist[1]) * ((*fields[i])(sites[i]+2+0) - (*fields[i])(sites[i]+0));
	gradxi[0] += (1.-ref_dist[1]) * ref_dist[2] * ((*fields[i])(sites[i]+2+0) - (*fields[i])(sites[i]+2));
	gradxi[1] += (1.-ref_dist[0]) * ref_dist[2] * ((*fields[i])(sites[i]+2+1) - (*fields[i])(sites[i]+2));
	gradxi[2] += (1.-ref_dist[0]) * ref_dist[1] * ((*fields[i])(sites[i]+2+1) - (*fields[i])(sites[i]+1));
	gradxi[0] += ref_dist[1] * ref_dist[2] * ((*fields[i])(sites[i]+2+1+0) - (*fields[i])(sites[i]+2+1));
	gradxi[1] += ref_dist[0] * ref_dist[2] * ((*fields[i])(sites[i]+2+1+0) - (*fields[i])(sites[i]+2+0));
	gradxi[2] += ref_dist[0] * ref_dist[1] * ((*fields[i])(sites[i]+2+1+0) - (*fields[i])(sites[i]+1+0));
	
	gradxi[0] /= lat_resolution;
	gradxi[1] /= lat_resolution;
	gradxi[2] /= lat_resolution;
	
	for (i = 0; i < 3; i++) ref_dist[i] += coeff*gradxi[i]/lat_resolution;
	
	gradxi[0] = (1.-ref_dist[1]) * (1.-ref_dist[2]) * ((*fields[i])(sites[i]+0) - (*fields[i])(sites[i]));
	gradxi[1] = (1.-ref_dist[0]) * (1.-ref_dist[2]) * ((*fields[i])(sites[i]+1) - (*fields[i])(sites[i]));
	gradxi[2] = (1.-ref_dist[0]) * (1.-ref_dist[1]) * ((*fields[i])(sites[i]+2) - (*fields[i])(sites[i]));
	gradxi[0] += ref_dist[1] * (1.-ref_dist[2]) * ((*fields[i])(sites[i]+1+0) - (*fields[i])(sites[i]+1));
	gradxi[1] += ref_dist[0] * (1.-ref_dist[2]) * ((*fields[i])(sites[i]+1+0) - (*fields[i])(sites[i]+0));
	gradxi[2] += ref_dist[0] * (1.-ref_dist[1]) * ((*fields[i])(sites[i]+2+0) - (*fields[i])(sites[i]+0));
	gradxi[0] += (1.-ref_dist[1]) * ref_dist[2] * ((*fields[i])(sites[i]+2+0) - (*fields[i])(sites[i]+2));
	gradxi[1] += (1.-ref_dist[0]) * ref_dist[2] * ((*fields[i])(sites[i]+2+1) - (*fields[i])(sites[i]+2));
	gradxi[2] += (1.-ref_dist[0]) * ref_dist[1] * ((*fields[i])(sites[i]+2+1) - (*fields[i])(sites[i]+1));
	gradxi[0] += ref_dist[1] * ref_dist[2] * ((*fields[i])(sites[i]+2+1+0) - (*fields[i])(sites[i]+2+1));
	gradxi[1] += ref_dist[0] * ref_dist[2] * ((*fields[i])(sites[i]+2+1+0) - (*fields[i])(sites[i]+2+0));
	gradxi[2] += ref_dist[0] * ref_dist[1] * ((*fields[i])(sites[i]+2+1+0) - (*fields[i])(sites[i]+1+0));
	
	gradxi[0] /= lat_resolution;
	gradxi[1] /= lat_resolution;
	gradxi[2] /= lat_resolution;
	
	if (noutputs > 0)
		*outputs = coeff * sqrt(gradxi[0]*gradxi[0] + gradxi[1]*gradxi[1] + gradxi[2]*gradxi[2]);

	for (i = 0; i < 3; i++) (*part).pos[i] += coeff*gradxi[i];
}


void generateIC_2ndorder(metadata & sim, icsettings & ic, cosmology & cosmo, const double fourpiG, Particles<part_simple,part_simple_info,part_simple_dataType> * pcls_cdm, Particles<part_simple,part_simple_info,part_simple_dataType> * pcls_b, Particles<part_simple,part_simple_info,part_simple_dataType> * pcls_ncdm, double * maxvel, Field<Real> * phi, Field<Real> * chi, Field<Real> * Bi, Field<Real> * source, Field<Real> * Sij, Field<Cplx> * scalarFT, Field<Cplx> * BiFT, Field<Cplx> * SijFT, PlanFFT<Cplx> * plan_phi, PlanFFT<Cplx> * plan_chi, PlanFFT<Cplx> * plan_Bi, PlanFFT<Cplx> * plan_source, PlanFFT<Cplx> * plan_Sij, parameter * params, int & numparam)
{
	int i, j, p;
	double a = 1. / (1. + sim.z_in);
	float * pcldata = NULL;
	gsl_spline * pkspline = NULL;
	gsl_spline * nbspline = NULL;
	gsl_spline * vnbspline = NULL;
	gsl_spline * tk_d1 = NULL;
	gsl_spline * tk_d2 = NULL;
	gsl_spline * tk_t1 = NULL;
	gsl_spline * tk_t2 = NULL;
	double * temp1 = NULL;
	double * temp2 = NULL;
	Site x(phi->lattice());
	rKSite kFT(scalarFT->lattice());
	double max_displacement;
	double rescale;
	double mean_q;
	part_simple_info pcls_cdm_info;
	part_simple_dataType pcls_cdm_dataType;
	part_simple_info pcls_b_info;
	part_simple_dataType pcls_b_dataType;
	part_simple_info pcls_ncdm_info[MAX_PCL_SPECIES];
	part_simple_dataType pcls_ncdm_dataType;
	Real boxSize[3] = {1.,1.,1.};
	char ncdm_name[8];
	Field<Real> * ic_fields[2];
	string filename;
	
	ic_fields[0] = chi;
	ic_fields[1] = phi;

#ifdef HAVE_CLASS
  	background class_background;
  	thermo class_thermo;
  	perturbs class_perturbs;
#endif
	
	loadHomogeneousTemplate(ic.pclfile[0], sim.numpcl[0], pcldata);
	
	if (pcldata == NULL)
	{
		COUT << " error: particle data was empty!" << endl;
		parallel.abortForce();
	}
	
/*	if (ic.flags & ICFLAG_CORRECT_DISPLACEMENT)
		generateCICKernel(*source, sim.numpcl[0], pcldata, ic.numtile[0]);
	else
		generateCICKernel(*source);	
	
	plan_source->execute(FFT_FORWARD);
	
	if (ic.pkfile[0] != '\0')	// initial displacements & velocities are derived from a single power spectrum
	{
		loadPowerSpectrum(ic.pkfile, pkspline, sim.boxsize);
	
		if (pkspline == NULL)
		{
			COUT << " error: power spectrum was empty!" << endl;
			parallel.abortForce();
		}
		
		temp1 = (double *) malloc(pkspline->size * sizeof(double));
		temp2 = (double *) malloc(pkspline->size * sizeof(double));
		
		for (i = 0; i < pkspline->size; i++)
		{
			temp1[i] = pkspline->x[i];
			temp2[i] = pkspline->y[i] / sim.boxsize / sim.boxsize;
		}
		gsl_spline_free(pkspline);
		pkspline = gsl_spline_alloc(gsl_interp_cspline, i);
		gsl_spline_init(pkspline, temp1, temp2, i);
	
		generateDisplacementField(*scalarFT, sim.gr_flag * Hconf(a, fourpiG, cosmo) * Hconf(a, fourpiG, cosmo), pkspline, (unsigned int) ic.seed, ic.flags & ICFLAG_KSPHERE);
	}
	else					// initial displacements and velocities are set by individual transfer functions
	{
#ifdef HAVE_CLASS
		if (ic.tkfile[0] == '\0')
		{
			initializeCLASSstructures(sim, ic, cosmo, class_background, class_thermo, class_perturbs, params, numparam);
			loadTransferFunctions(class_background, class_perturbs, tk_d1, tk_t1, "tot", sim.boxsize, sim.z_in, cosmo.h);
		}
		else
#endif
		loadTransferFunctions(ic.tkfile, tk_d1, tk_t1, "tot", sim.boxsize, cosmo.h);
		
		if (tk_d1 == NULL || tk_t1 == NULL)
		{
			COUT << " error: total transfer function was empty!" << endl;
			parallel.abortForce();
		}
		
		temp1 = (double *) malloc(tk_d1->size * sizeof(double));
		temp2 = (double *) malloc(tk_d1->size * sizeof(double));
		
		rescale = 3. * Hconf(a, fourpiG, cosmo) * Hconf(a, fourpiG, cosmo) * Hconf(a, fourpiG, cosmo) * (1. + 0.5 * Hconf(a, fourpiG, cosmo) * Hconf(a, fourpiG, cosmo) * ((1. / Hconf(0.98 * a, fourpiG, cosmo) / Hconf(0.98 * a, fourpiG, cosmo)) - (8. / Hconf(0.99 * a, fourpiG, cosmo) / Hconf(0.99 * a, fourpiG, cosmo)) + (8. / Hconf(1.01 * a, fourpiG, cosmo) / Hconf(1.01 * a, fourpiG, cosmo)) - (1. / Hconf(1.02 * a, fourpiG, cosmo) / Hconf(1.02 * a, fourpiG, cosmo))) / 0.12);
		for (i = 0; i < tk_d1->size; i++) // construct phi
			temp1[i] = (1.5 * (Hconf(a, fourpiG, cosmo) * Hconf(a, fourpiG, cosmo) - Hconf(1., fourpiG, cosmo) * Hconf(1., fourpiG, cosmo) * a * a * cosmo.Omega_Lambda) * tk_d1->y[i] + rescale * tk_t1->y[i] / tk_d1->x[i] / tk_d1->x[i]) * M_PI * sqrt(Pk_primordial(tk_d1->x[i] * cosmo.h / sim.boxsize, ic) / tk_d1->x[i]) / tk_d1->x[i];

		if (sim.gr_flag == 0)
		{
			for (i = 0; i < tk_t1->size; i++) // construct gauge correction for N-body gauge (3 Hconf theta_tot / k^2)
				temp2[i] = -3. * Hconf(a, fourpiG, cosmo)  * M_PI * tk_t1->y[i] * sqrt(Pk_primordial(tk_d1->x[i] * cosmo.h / sim.boxsize, ic) / tk_d1->x[i]) / tk_d1->x[i] / tk_d1->x[i] / tk_d1->x[i];

			nbspline = gsl_spline_alloc(gsl_interp_cspline, tk_t1->size);
			gsl_spline_init(nbspline, tk_t1->x, temp2, tk_t1->size);
		}
		
		pkspline = gsl_spline_alloc(gsl_interp_cspline, tk_d1->size);
		gsl_spline_init(pkspline, tk_d1->x, temp1, tk_d1->size);
		gsl_spline_free(tk_d1);
		gsl_spline_free(tk_t1);

#ifdef HAVE_CLASS
		if (ic.tkfile[0] == '\0')
		{
			if (sim.gr_flag == 0)
			{
				loadTransferFunctions(class_background, class_perturbs, tk_d1, tk_t1, NULL, sim.boxsize, sim.z_in, cosmo.h);

				for (i = 0; i < tk_d1->size; i++)
					temp1[i] = -tk_d1->y[i];

				gsl_spline_free(tk_d1);
				gsl_spline_free(tk_t1);

				loadTransferFunctions(class_background, class_perturbs, tk_d1, tk_t1, NULL, sim.boxsize, (sim.z_in + 0.01) / 0.99, cosmo.h);

				for (i = 0; i < tk_d1->size; i++)
					temp1[i] += tk_d1->y[i];

				gsl_spline_free(tk_d1);
				gsl_spline_free(tk_t1);

				loadTransferFunctions(class_background, class_perturbs, tk_d1, tk_t1, "tot", sim.boxsize, (sim.z_in + 0.01) / 0.99, cosmo.h);

				for (i = 0; i < tk_d1->size; i++) // construct gauge correction for N-body gauge velocities
					temp1[i] = -99.5 * Hconf(0.995 * a, fourpiG, cosmo) * (3. * temp1[i] * M_PI * sqrt(Pk_primordial(tk_d1->x[i] * cosmo.h / sim.boxsize, ic) / tk_d1->x[i]) / tk_d1->x[i] + (temp2[i] + 3. * Hconf(0.99 * a, fourpiG, cosmo)  * M_PI * tk_t1->y[i] * sqrt(Pk_primordial(tk_d1->x[i] * cosmo.h / sim.boxsize, ic) / tk_d1->x[i]) / tk_d1->x[i] / tk_d1->x[i] / tk_d1->x[i]));

				vnbspline = gsl_spline_alloc(gsl_interp_cspline, tk_t1->size);
				gsl_spline_init(vnbspline, tk_t1->x, temp1, tk_t1->size);

				gsl_spline_free(tk_d1);
				gsl_spline_free(tk_t1);
			}

			loadTransferFunctions(class_background, class_perturbs, tk_d1, tk_t1, "cdm", sim.boxsize, sim.z_in, cosmo.h);
		}
		else
#endif		
		loadTransferFunctions(ic.tkfile, tk_d1, tk_t1, "cdm", sim.boxsize, cosmo.h);	// get transfer functions for CDM
		
		if (tk_d1 == NULL || tk_t1 == NULL)
		{
			COUT << " error: cdm transfer function was empty!" << endl;
			parallel.abortForce();
		}
		
		if (sim.baryon_flag > 0)
		{
#ifdef HAVE_CLASS
			if (ic.tkfile[0] == '\0')
				loadTransferFunctions(class_background, class_perturbs, tk_d2, tk_t2, "b", sim.boxsize, sim.z_in, cosmo.h);
			else
#endif
			loadTransferFunctions(ic.tkfile, tk_d2, tk_t2, "b", sim.boxsize, cosmo.h);	// get transfer functions for baryons
		
			if (tk_d2 == NULL || tk_t2 == NULL)
			{
				COUT << " error: baryon transfer function was empty!" << endl;
				parallel.abortForce();
			}
			if (tk_d2->size != tk_d1->size)
			{
				COUT << " error: baryon transfer function line number mismatch!" << endl;
				parallel.abortForce();
			}
		}
		
		if (sim.baryon_flag == 2)	// baryon treatment = blend; compute displacement & velocity from weighted average
		{
			if (sim.gr_flag > 0)
			{
				for (i = 0; i < tk_d1->size; i++)
					temp1[i] = -3. * pkspline->y[i] / pkspline->x[i] / pkspline->x[i] - ((cosmo.Omega_cdm * tk_d1->y[i] + cosmo.Omega_b * tk_d2->y[i]) / (cosmo.Omega_cdm + cosmo.Omega_b)) * M_PI * sqrt(Pk_primordial(tk_d1->x[i] * cosmo.h / sim.boxsize, ic) / tk_d1->x[i]) / tk_d1->x[i];
			}
			else
			{
				for (i = 0; i < tk_d1->size; i++)
					temp1[i] = nbspline->y[i] - ((cosmo.Omega_cdm * tk_d1->y[i] + cosmo.Omega_b * tk_d2->y[i]) / (cosmo.Omega_cdm + cosmo.Omega_b)) * M_PI * sqrt(Pk_primordial(tk_d1->x[i] * cosmo.h / sim.boxsize, ic) / tk_d1->x[i]) / tk_d1->x[i];
			}
			if (sim.gr_flag > 0 || vnbspline == NULL)
			{
				for (i = 0; i < tk_d1->size; i++)
					temp2[i] = -a * ((cosmo.Omega_cdm * tk_t1->y[i] + cosmo.Omega_b * tk_t2->y[i]) / (cosmo.Omega_cdm + cosmo.Omega_b)) * M_PI * sqrt(Pk_primordial(tk_d1->x[i] * cosmo.h / sim.boxsize, ic) / tk_d1->x[i]) / tk_d1->x[i];
			}
			else
			{
				for (i = 0; i < tk_d1->size; i++)
					temp2[i] = a * vnbspline->y[i] - a * ((cosmo.Omega_cdm * tk_t1->y[i] + cosmo.Omega_b * tk_t2->y[i]) / (cosmo.Omega_cdm + cosmo.Omega_b)) * M_PI * sqrt(Pk_primordial(tk_d1->x[i] * cosmo.h / sim.boxsize, ic) / tk_d1->x[i]) / tk_d1->x[i];
			}

			gsl_spline_free(tk_d1);
			gsl_spline_free(tk_t1);
			tk_d1 = gsl_spline_alloc(gsl_interp_cspline, tk_d2->size);
			tk_t1 = gsl_spline_alloc(gsl_interp_cspline, tk_d2->size);
			gsl_spline_init(tk_d1, tk_d2->x, temp1, tk_d2->size);
			gsl_spline_init(tk_t1, tk_d2->x, temp2, tk_d2->size);
			gsl_spline_free(tk_d2);
			gsl_spline_free(tk_t2);
		}
		
		if (sim.baryon_flag == 3)	// baryon treatment = hybrid; compute displacement & velocity from weighted average (sub-species)
		{
			if (8. * cosmo.Omega_b / (cosmo.Omega_cdm + cosmo.Omega_b) > 1.)
			{
				if (sim.gr_flag > 0)
				{
					for (i = 0; i < tk_d1->size; i++)
						temp1[i] = -3. * pkspline->y[i] / pkspline->x[i] / pkspline->x[i] - ((8. * cosmo.Omega_cdm * tk_d1->y[i] + (7. * cosmo.Omega_b - cosmo.Omega_cdm) * tk_d2->y[i]) / (cosmo.Omega_cdm + cosmo.Omega_b) / 7.) * M_PI * sqrt(Pk_primordial(tk_d1->x[i] * cosmo.h / sim.boxsize, ic) / tk_d1->x[i]) / tk_d1->x[i];
				}
				else
				{
					for (i = 0; i < tk_d1->size; i++)
						temp1[i] = nbspline->y[i] - ((8. * cosmo.Omega_cdm * tk_d1->y[i] + (7. * cosmo.Omega_b - cosmo.Omega_cdm) * tk_d2->y[i]) / (cosmo.Omega_cdm + cosmo.Omega_b) / 7.) * M_PI * sqrt(Pk_primordial(tk_d1->x[i] * cosmo.h / sim.boxsize, ic) / tk_d1->x[i]) / tk_d1->x[i];
				}
				if (sim.gr_flag > 0 || vnbspline == NULL)
				{
					for (i = 0; i < tk_d1->size; i++)
						temp2[i] = -a * ((8. * cosmo.Omega_cdm * tk_t1->y[i] + (7. * cosmo.Omega_b - cosmo.Omega_cdm) * tk_t2->y[i]) / (cosmo.Omega_cdm + cosmo.Omega_b) / 7.) * M_PI * sqrt(Pk_primordial(tk_d1->x[i] * cosmo.h / sim.boxsize, ic) / tk_d1->x[i]) / tk_d1->x[i];
				}
				else
				{
					for (i = 0; i < tk_d1->size; i++)
						temp2[i] = a * vnbspline->y[i] - a * ((8. * cosmo.Omega_cdm * tk_t1->y[i] + (7. * cosmo.Omega_b - cosmo.Omega_cdm) * tk_t2->y[i]) / (cosmo.Omega_cdm + cosmo.Omega_b) / 7.) * M_PI * sqrt(Pk_primordial(tk_d1->x[i] * cosmo.h / sim.boxsize, ic) / tk_d1->x[i]) / tk_d1->x[i];
				}

				gsl_spline_free(tk_d1);
				gsl_spline_free(tk_t1);
				tk_d1 = gsl_spline_alloc(gsl_interp_cspline, tk_d2->size);
				tk_t1 = gsl_spline_alloc(gsl_interp_cspline, tk_d2->size);
				gsl_spline_init(tk_d1, tk_d2->x, temp1, tk_d2->size);
				gsl_spline_init(tk_t1, tk_d2->x, temp2, tk_d2->size);
			}
			else
			{
				if (sim.gr_flag > 0)
				{
					for (i = 0; i < tk_d1->size; i++)
						temp1[i] = -3. * pkspline->y[i] / pkspline->x[i] / pkspline->x[i] - (((cosmo.Omega_cdm - 7. * cosmo.Omega_b) * tk_d1->y[i] + 8. * cosmo.Omega_b * tk_d2->y[i]) / (cosmo.Omega_cdm + cosmo.Omega_b)) * M_PI * sqrt(Pk_primordial(tk_d1->x[i] * cosmo.h / sim.boxsize, ic) / tk_d1->x[i]) / tk_d1->x[i];
				}
				else
				{
					for (i = 0; i < tk_d1->size; i++)
						temp1[i] = nbspline->y[i] - (((cosmo.Omega_cdm - 7. * cosmo.Omega_b) * tk_d1->y[i] + 8. * cosmo.Omega_b * tk_d2->y[i]) / (cosmo.Omega_cdm + cosmo.Omega_b)) * M_PI * sqrt(Pk_primordial(tk_d1->x[i] * cosmo.h / sim.boxsize, ic) / tk_d1->x[i]) / tk_d1->x[i];
				}
				if (sim.gr_flag > 0 || vnbspline == NULL)
				{
					for (i = 0; i < tk_d1->size; i++)
						temp2[i] = -a * (((cosmo.Omega_cdm - 7. * cosmo.Omega_b) * tk_t1->y[i] + 8. * cosmo.Omega_b * tk_t2->y[i]) / (cosmo.Omega_cdm + cosmo.Omega_b)) * M_PI * sqrt(Pk_primordial(tk_d1->x[i] * cosmo.h / sim.boxsize, ic) / tk_d1->x[i]) / tk_d1->x[i];
				}
				else
				{
					for (i = 0; i < tk_d1->size; i++)
						temp2[i] = a * vnbspline->y[i] - a * (((cosmo.Omega_cdm - 7. * cosmo.Omega_b) * tk_t1->y[i] + 8. * cosmo.Omega_b * tk_t2->y[i]) / (cosmo.Omega_cdm + cosmo.Omega_b)) * M_PI * sqrt(Pk_primordial(tk_d1->x[i] * cosmo.h / sim.boxsize, ic) / tk_d1->x[i]) / tk_d1->x[i];
				}

				gsl_spline_free(tk_d2);
				gsl_spline_free(tk_t2);
				tk_d2 = gsl_spline_alloc(gsl_interp_cspline, tk_d1->size);
				tk_t2 = gsl_spline_alloc(gsl_interp_cspline, tk_d1->size);
				gsl_spline_init(tk_d2, tk_d2->x, temp1, tk_d1->size);
				gsl_spline_init(tk_t2, tk_d2->x, temp2, tk_d1->size);
			}
		}
		
		if (sim.baryon_flag == 1 || (sim.baryon_flag == 3 && 8. * cosmo.Omega_b / (cosmo.Omega_cdm + cosmo.Omega_b) > 1.)) // compute baryonic displacement & velocity
		{
			if (sim.gr_flag > 0)
			{
				for (i = 0; i < tk_d1->size; i++)
					temp1[i] = -3. * pkspline->y[i] / pkspline->x[i] / pkspline->x[i] - tk_d2->y[i] * M_PI * sqrt(Pk_primordial(tk_d2->x[i] * cosmo.h / sim.boxsize, ic) / tk_d2->x[i]) / tk_d2->x[i];
			}
			else
			{
				for (i = 0; i < tk_d1->size; i++)
					temp1[i] = nbspline->y[i] - tk_d2->y[i] * M_PI * sqrt(Pk_primordial(tk_d2->x[i] * cosmo.h / sim.boxsize, ic) / tk_d2->x[i]) / tk_d2->x[i];
			}
			if (sim.gr_flag > 0 || vnbspline == NULL)
			{
				for (i = 0; i < tk_d1->size; i++)
					temp2[i] = -a * tk_t2->y[i] * M_PI * sqrt(Pk_primordial(tk_d2->x[i] * cosmo.h / sim.boxsize, ic) / tk_d2->x[i]) / tk_d2->x[i];
			}
			else
			{
				for (i = 0; i < tk_d1->size; i++)
					temp2[i] = a * vnbspline->y[i] - a * tk_t2->y[i] * M_PI * sqrt(Pk_primordial(tk_d2->x[i] * cosmo.h / sim.boxsize, ic) / tk_d2->x[i]) / tk_d2->x[i];
			}

			gsl_spline_free(tk_d2);
			gsl_spline_free(tk_t2);
			tk_d2 = gsl_spline_alloc(gsl_interp_cspline, tk_d1->size);
			tk_t2 = gsl_spline_alloc(gsl_interp_cspline, tk_d1->size);
			gsl_spline_init(tk_d2, tk_d1->x, temp1, tk_d1->size);
			gsl_spline_init(tk_t2, tk_d1->x, temp2, tk_d1->size);
		}
		
		if (sim.baryon_flag < 2 || (sim.baryon_flag == 3 && 8. * cosmo.Omega_b / (cosmo.Omega_cdm + cosmo.Omega_b) <= 1.))	// compute CDM displacement & velocity
		{
			if (sim.gr_flag > 0)
			{
				for (i = 0; i < tk_d1->size; i++)
					temp1[i] = -3. * pkspline->y[i] / pkspline->x[i] / pkspline->x[i] - tk_d1->y[i] * M_PI * sqrt(Pk_primordial(tk_d1->x[i] * cosmo.h / sim.boxsize, ic) / tk_d1->x[i]) / tk_d1->x[i];
			}
			else
			{
				for (i = 0; i < tk_d1->size; i++)
					temp1[i] = nbspline->y[i] - tk_d1->y[i] * M_PI * sqrt(Pk_primordial(tk_d1->x[i] * cosmo.h / sim.boxsize, ic) / tk_d1->x[i]) / tk_d1->x[i];
			}
			if (sim.gr_flag > 0 || vnbspline == NULL)
			{	
				for (i = 0; i < tk_d1->size; i++)
					temp2[i] = -a * tk_t1->y[i] * M_PI * sqrt(Pk_primordial(tk_d1->x[i] * cosmo.h / sim.boxsize, ic) / tk_d1->x[i]) / tk_d1->x[i];
			}
			else
			{
				for (i = 0; i < tk_d1->size; i++)
					temp2[i] = a * vnbspline->y[i] - a * tk_t1->y[i] * M_PI * sqrt(Pk_primordial(tk_d1->x[i] * cosmo.h / sim.boxsize, ic) / tk_d1->x[i]) / tk_d1->x[i];
			}

			gsl_spline_free(tk_d1);
			gsl_spline_free(tk_t1);
			tk_d1 = gsl_spline_alloc(gsl_interp_cspline, pkspline->size);
			tk_t1 = gsl_spline_alloc(gsl_interp_cspline, pkspline->size);
			gsl_spline_init(tk_d1, pkspline->x, temp1, pkspline->size);
			gsl_spline_init(tk_t1, pkspline->x, temp2, pkspline->size);
		}
		
		if ((sim.baryon_flag == 1 && !(ic.flags & ICFLAG_CORRECT_DISPLACEMENT)) || sim.baryon_flag == 3)
		{
			generateDisplacementField(*scalarFT, 0., tk_d2, (unsigned int) ic.seed, ic.flags & ICFLAG_KSPHERE);
			gsl_spline_free(tk_d2);
			plan_phi->execute(FFT_BACKWARD);
			phi->updateHalo();	// phi now contains the baryonic displacement
			plan_source->execute(FFT_FORWARD);
		}
		
		generateDisplacementField(*scalarFT, 0., tk_d1, (unsigned int) ic.seed, ic.flags & ICFLAG_KSPHERE);
		gsl_spline_free(tk_d1);
	}
		
	plan_chi->execute(FFT_BACKWARD);
	chi->updateHalo();	// chi now contains the CDM displacement */
	
	filename.assign(ic.displacementfile);
	chi->loadHDF5(filename);
	chi->updateHalo();
	phi->loadHDF5(filename);
	phi->updateHalo();
	
	strcpy(pcls_cdm_info.type_name, "part_simple");
	if (sim.baryon_flag == 1)
		pcls_cdm_info.mass = cosmo.Omega_cdm / (Real) (sim.numpcl[0]*(long)ic.numtile[0]*(long)ic.numtile[0]*(long)ic.numtile[0]);
	else
		pcls_cdm_info.mass = (cosmo.Omega_cdm + cosmo.Omega_b) / (Real) (sim.numpcl[0]*(long)ic.numtile[0]*(long)ic.numtile[0]*(long)ic.numtile[0]);
	pcls_cdm_info.relativistic = false;
	
	pcls_cdm->initialize(pcls_cdm_info, pcls_cdm_dataType, &(phi->lattice()), boxSize);
	
	initializeParticlePositions(sim.numpcl[0], pcldata, ic.numtile[0], *pcls_cdm);
	i = MAX;
	if (sim.baryon_flag == 3)	// baryon treatment = hybrid; displace particles using both displacement fields
		pcls_cdm->moveParticles(displace_pcls_ic_2ndorder, 1., ic_fields, 2, NULL, &max_displacement, &i, 1);
	else
		pcls_cdm->moveParticles(displace_pcls_ic_2ndorder, 1., &chi, 1, NULL, &max_displacement, &i, 1);	// displace CDM particles
	
	sim.numpcl[0] *= (long) ic.numtile[0] * (long) ic.numtile[0] * (long) ic.numtile[0];
	
	COUT << " " << sim.numpcl[0] << " cdm particles initialized: maximum displacement = " << max_displacement * sim.numpts << " lattice units." << endl;
	
	free(pcldata);
	
	if (sim.baryon_flag == 1)
	{
		loadHomogeneousTemplate(ic.pclfile[1], sim.numpcl[1], pcldata);
	
		if (pcldata == NULL)
		{
			COUT << " error: particle data was empty!" << endl;
			parallel.abortForce();
		}
		
		/*if (ic.flags & ICFLAG_CORRECT_DISPLACEMENT)
		{
			generateCICKernel(*phi, sim.numpcl[1], pcldata, ic.numtile[1]);
			plan_phi->execute(FFT_FORWARD);
			generateDisplacementField(*scalarFT, 0., tk_d2, (unsigned int) ic.seed, ic.flags & ICFLAG_KSPHERE);
			gsl_spline_free(tk_d2);
			plan_phi->execute(FFT_BACKWARD);
			phi->updateHalo();
		}*/
		
		strcpy(pcls_b_info.type_name, "part_simple");
		pcls_b_info.mass = cosmo.Omega_b / (Real) (sim.numpcl[1]*(long)ic.numtile[1]*(long)ic.numtile[1]*(long)ic.numtile[1]);
		pcls_b_info.relativistic = false;
	
		pcls_b->initialize(pcls_b_info, pcls_b_dataType, &(phi->lattice()), boxSize);
	
		initializeParticlePositions(sim.numpcl[1], pcldata, ic.numtile[1], *pcls_b);
		i = MAX;
		pcls_b->moveParticles(displace_pcls_ic_2ndorder, 1., &phi, 1, NULL, &max_displacement, &i, 1);	// displace baryon particles
	
		sim.numpcl[1] *= (long) ic.numtile[1] * (long) ic.numtile[1] * (long) ic.numtile[1];
	
		COUT << " " << sim.numpcl[1] << " baryon particles initialized: maximum displacement = " << max_displacement * sim.numpts << " lattice units." << endl;
	
		free(pcldata);
	}
	
	if (ic.pkfile[0] == '\0')	// set velocities using transfer functions
	{
		/*if (ic.flags & ICFLAG_CORRECT_DISPLACEMENT)
			generateCICKernel(*source);
		
		plan_source->execute(FFT_FORWARD);
		
		if (sim.baryon_flag == 1 || sim.baryon_flag == 3)
		{
			generateDisplacementField(*scalarFT, 0., tk_t2, (unsigned int) ic.seed, ic.flags & ICFLAG_KSPHERE, 0);
			plan_phi->execute(FFT_BACKWARD);
			phi->updateHalo();	// phi now contains the baryonic velocity potential
			gsl_spline_free(tk_t2);
			plan_source->execute(FFT_FORWARD);
		}
		
		generateDisplacementField(*scalarFT, 0., tk_t1, (unsigned int) ic.seed, ic.flags & ICFLAG_KSPHERE, 0);
		plan_chi->execute(FFT_BACKWARD);
		chi->updateHalo();	// chi now contains the CDM velocity potential
		gsl_spline_free(tk_t1);		*/
		
		filename.assign(ic.velocityfile[0]);
		phi->loadHDF5(filename);
		chi->loadHDF5(filename);
		phi->updateHalo();
		chi->updateHalo();	
		
		if (sim.baryon_flag == 3)	// baryon treatment = hybrid; set velocities using both velocity potentials
			maxvel[0] = pcls_cdm->updateVel(initialize_q_ic_basic, 1., ic_fields, 2) / a;
		else
			maxvel[0] = pcls_cdm->updateVel(initialize_q_ic_basic, 1., &chi, 1) / a;	// set CDM velocities
		
		if (sim.baryon_flag == 1)
			maxvel[1] = pcls_b->updateVel(initialize_q_ic_basic, 1., &phi, 1) / a;	// set baryon velocities
	}
	
	if (sim.baryon_flag > 1) sim.baryon_flag = 0;
	
/*	for (p = 0; p < cosmo.num_ncdm; p++)	// initialization of non-CDM species
	{
		if (ic.numtile[1+sim.baryon_flag+p] < 1) continue;

		loadHomogeneousTemplate(ic.pclfile[1+sim.baryon_flag+p], sim.numpcl[1+sim.baryon_flag+p], pcldata);
	
		if (pcldata == NULL)
		{
			COUT << " error: particle data was empty!" << endl;
			parallel.abortForce();
		}
		
		if (ic.pkfile[0] == '\0')
		{
			sprintf(ncdm_name, "ncdm[%d]", p);
#ifdef HAVE_CLASS
			if (ic.tkfile[0] == '\0')
				loadTransferFunctions(class_background, class_perturbs, tk_d1, tk_t1, ncdm_name, sim.boxsize, sim.z_in, cosmo.h);
			else
#endif
			loadTransferFunctions(ic.tkfile, tk_d1, tk_t1, ncdm_name, sim.boxsize, cosmo.h);
		
			if (tk_d1 == NULL || tk_t1 == NULL)
			{
				COUT << " error: ncdm transfer function was empty! (species " << p << ")" << endl;
				parallel.abortForce();
			}
			
			if (sim.gr_flag > 0)
			{
				for (i = 0; i < tk_d1->size; i++)
					temp1[i] = -3. * pkspline->y[i] / pkspline->x[i] / pkspline->x[i] - tk_d1->y[i] * M_PI * sqrt(Pk_primordial(tk_d1->x[i] * cosmo.h / sim.boxsize, ic) / tk_d1->x[i]) / tk_d1->x[i];
			}
			else
			{
				for (i = 0; i < tk_d1->size; i++)
					temp1[i] = nbspline->y[i] - tk_d1->y[i] * M_PI * sqrt(Pk_primordial(tk_d1->x[i] * cosmo.h / sim.boxsize, ic) / tk_d1->x[i]) / tk_d1->x[i];
			}
			if (sim.gr_flag > 0 || vnbspline == NULL)
			{
				for (i = 0; i < tk_d1->size; i++)
					temp2[i] = -a * tk_t1->y[i] * M_PI * sqrt(Pk_primordial(tk_d1->x[i] * cosmo.h / sim.boxsize, ic) / tk_d1->x[i]) / tk_d1->x[i];
			}
			else
			{
				for (i = 0; i < tk_d1->size; i++)
					temp2[i] = a * vnbspline->y[i] - a * tk_t1->y[i] * M_PI * sqrt(Pk_primordial(tk_d1->x[i] * cosmo.h / sim.boxsize, ic) / tk_d1->x[i]) / tk_d1->x[i];
			}

			gsl_spline_free(tk_d1);
			gsl_spline_free(tk_t1);
			tk_d1 = gsl_spline_alloc(gsl_interp_cspline, pkspline->size);
			tk_t1 = gsl_spline_alloc(gsl_interp_cspline, pkspline->size);
			gsl_spline_init(tk_d1, pkspline->x, temp1, pkspline->size);
			gsl_spline_init(tk_t1, pkspline->x, temp2, pkspline->size);
			
			plan_source->execute(FFT_FORWARD);
			generateDisplacementField(*scalarFT, 0., tk_d1, (unsigned int) ic.seed, ic.flags & ICFLAG_KSPHERE);
			plan_chi->execute(FFT_BACKWARD);	// chi now contains the displacement for the non-CDM species
			chi->updateHalo();
			gsl_spline_free(tk_d1);
			
			plan_source->execute(FFT_FORWARD);
			generateDisplacementField(*scalarFT, 0., tk_t1, (unsigned int) ic.seed, ic.flags & ICFLAG_KSPHERE, 0);
			plan_phi->execute(FFT_BACKWARD);	// phi now contains the velocity potential for the non-CDM species
			phi->updateHalo();
			gsl_spline_free(tk_t1);
		}
		
		strcpy(pcls_ncdm_info[p].type_name, "part_simple");
		pcls_ncdm_info[p].mass = cosmo.Omega_ncdm[p] / (Real) (sim.numpcl[1+sim.baryon_flag+p]*(long)ic.numtile[1+sim.baryon_flag+p]*(long)ic.numtile[1+sim.baryon_flag+p]*(long)ic.numtile[1+sim.baryon_flag+p]);
		pcls_ncdm_info[p].relativistic = true;
		
		pcls_ncdm[p].initialize(pcls_ncdm_info[p], pcls_ncdm_dataType, &(phi->lattice()), boxSize);
		
		initializeParticlePositions(sim.numpcl[1+sim.baryon_flag+p], pcldata, ic.numtile[1+sim.baryon_flag+p], pcls_ncdm[p]);
		i = MAX;
		pcls_ncdm[p].moveParticles(displace_pcls_ic_basic, 1., &chi, 1, NULL, &max_displacement, &i, 1);	// displace non-CDM particles
		
		sim.numpcl[1+sim.baryon_flag+p] *= (long) ic.numtile[1+sim.baryon_flag+p] * (long) ic.numtile[1+sim.baryon_flag+p] * (long) ic.numtile[1+sim.baryon_flag+p];
	
		COUT << " " << sim.numpcl[1+sim.baryon_flag+p] << " ncdm particles initialized for species " << p+1 << ": maximum displacement = " << max_displacement * sim.numpts << " lattice units." << endl;
		
		free(pcldata);
		
		if (ic.pkfile[0] == '\0')	// set non-CDM velocities using transfer functions
			pcls_ncdm[p].updateVel(initialize_q_ic_basic, 1., &phi, 1);
	}

	free(temp1);
	free(temp2);
	
	if (ic.pkfile[0] == '\0')
	{
		plan_source->execute(FFT_FORWARD);
		generateDisplacementField(*scalarFT, 0., pkspline, (unsigned int) ic.seed, ic.flags & ICFLAG_KSPHERE, 0);
#ifdef HAVE_CLASS
		if (ic.tkfile[0] == '\0')
			freeCLASSstructures(class_background, class_thermo, class_perturbs);
#endif
	}
	else
	{
		projection_init(source);
		scalarProjectionCIC_project(pcls_cdm, source);
		if (sim.baryon_flag)
			scalarProjectionCIC_project(pcls_b, source);
		for (p = 0; p < cosmo.num_ncdm; p++)
		{
			if (ic.numtile[1+sim.baryon_flag+p] < 1) continue;
			scalarProjectionCIC_project(pcls_ncdm+p, source);
		}
		scalarProjectionCIC_comm(source);
	
		plan_source->execute(FFT_FORWARD);
	
		kFT.first();
		if (kFT.coord(0) == 0 && kFT.coord(1) == 0 && kFT.coord(2) == 0)
			(*scalarFT)(kFT) = Cplx(0.,0.);
				
		solveModifiedPoissonFT(*scalarFT, *scalarFT, fourpiG / a, 3. * sim.gr_flag * (Hconf(a, fourpiG, cosmo) * Hconf(a, fourpiG, cosmo) + fourpiG * cosmo.Omega_m / a));
	}
	
	plan_phi->execute(FFT_BACKWARD);
	phi->updateHalo();	// phi now finally contains phi
	
	if (ic.pkfile[0] != '\0')	// if power spectrum is used instead of transfer functions, set velocities using linear approximation
	{	
		rescale = a / Hconf(a, fourpiG, cosmo) / (1.5 * Omega_m(a, cosmo) + 2. * Omega_rad(a, cosmo));
		maxvel[0] = pcls_cdm->updateVel(initialize_q_ic_basic, rescale, &phi, 1) / a;
		if (sim.baryon_flag)
			maxvel[1] = pcls_b->updateVel(initialize_q_ic_basic, rescale, &phi, 1) / a;
	}
			
	for (p = 0; p < cosmo.num_ncdm; p++)
	{
		if (ic.numtile[1+sim.baryon_flag+p] < 1)
		{
			maxvel[1+sim.baryon_flag+p] = 0;
			continue;
		}

		if (ic.pkfile[0] != '\0') // if power spectrum is used instead of transfer functions, set bulk velocities using linear approximation
		{		
			rescale = a / Hconf(a, fourpiG, cosmo) / (1.5 * Omega_m(a, cosmo) + Omega_rad(a, cosmo));
			pcls_ncdm[p].updateVel(initialize_q_ic_basic, rescale, &phi, 1);
		}
		
		if (cosmo.m_ncdm[p] > 0.) // add velocity dispersion for non-CDM species
		{
			rescale = pow(cosmo.Omega_g * cosmo.h * cosmo.h / C_PLANCK_LAW, 0.25) * cosmo.T_ncdm[p] * C_BOLTZMANN_CST / cosmo.m_ncdm[p];
			mean_q = applyMomentumDistribution(pcls_ncdm+p, (unsigned int) (ic.seed + p), rescale);
			parallel.sum(mean_q);
			COUT << " species " << p+1 << " Fermi-Dirac distribution had mean q/m = " << mean_q / sim.numpcl[1+sim.baryon_flag+p] << endl;
		}
		maxvel[1+sim.baryon_flag+p] = pcls_ncdm[p].updateVel(update_q, 0., &phi, 1, &a);
	} */
	
	if (sim.gr_flag > 0 && ic.metricfile[0][0] != '\0')
	{
		filename.assign(ic.metricfile[0]);
		phi->loadHDF5(filename);
	}
	else
	{
		projection_init(source);
		scalarProjectionCIC_project(pcls_cdm, source);
		if (sim.baryon_flag)
			scalarProjectionCIC_project(pcls_b, source);
		scalarProjectionCIC_comm(source);
	
		plan_source->execute(FFT_FORWARD);
	
		kFT.first();
		if (kFT.coord(0) == 0 && kFT.coord(1) == 0 && kFT.coord(2) == 0)
			(*scalarFT)(kFT) = Cplx(0.,0.);
				
		solveModifiedPoissonFT(*scalarFT, *scalarFT, fourpiG / a, 3. * sim.gr_flag * (Hconf(a, fourpiG, cosmo) * Hconf(a, fourpiG, cosmo) + fourpiG * cosmo.Omega_m / a));
		plan_phi->execute(FFT_BACKWARD);
	}

	phi->updateHalo();
	
	projection_init(Bi);
	projection_T0i_project(pcls_cdm, Bi, phi);
	if (sim.baryon_flag)
		projection_T0i_project(pcls_b, Bi, phi);
	projection_T0i_comm(Bi);
	plan_Bi->execute(FFT_FORWARD);
	projectFTvector(*BiFT, *BiFT, fourpiG / (double) sim.numpts / (double) sim.numpts);	
	plan_Bi->execute(FFT_BACKWARD);	
	Bi->updateHalo();	// B initialized
	
	projection_init(Sij);
	projection_Tij_project(pcls_cdm, Sij, a, phi);
	if (sim.baryon_flag)
		projection_Tij_project(pcls_b, Sij, a, phi);
	projection_Tij_comm(Sij);
	
	prepareFTsource<Real>(*phi, *Sij, *Sij, 2. * fourpiG / a / (double) sim.numpts / (double) sim.numpts);	
	plan_Sij->execute(FFT_FORWARD);	
	projectFTscalar(*SijFT, *scalarFT);
	plan_chi->execute(FFT_BACKWARD);		
	chi->updateHalo();	// chi now finally contains chi

	gsl_spline_free(pkspline);
	if (sim.gr_flag == 0)
		gsl_spline_free(nbspline);
	if (vnbspline != NULL)
		gsl_spline_free(vnbspline);
}

#endif
