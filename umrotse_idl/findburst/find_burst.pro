function find_burst,oldmatch,mindelta,minsig,minchisq=minchisq,refra=refra,refdec=refdec,radius=radius,objid=objid, rotse=experiment, log=log,emask=emask,rmask=rmask
;+
; NAME:	find_burst
;
; CALLING SEQUENCE:	var = find_burst(match, mindelta, minsig)
;
; INPUTS:
;	match = match structure produced by make_match_struct
;	mindelta = threshold for total variation
;	minsig = number of stddev (stat.+sys.) for variation
;
; Keywords:
;	minchisq = minimum clipped chis-squared of good observations
;	emask = mask for extraction flags
;	rmask = mask for ROTSE observation flags
;       rotse = number of ROTSE experiment (= 1 or 3)
;	log = toggles whether to produce output text and postscript
;
; Return Value:	structure containing summary information on transient candidates
;
; PROCEDURE:	Searches for brief optical transients in a matched object
;		list from ROTSE trigger response data.
;
; Created:  4-23-99  Bob Kehoe
; Updated:  9-23-99  Bob Kehoe
; Updated: 11-10-99  Bob Kehoe
; Updated: 00-04-14  Bob Kehoe -- added relative photometry correction 
; Updated: 05-31-00  Bob Kehoe -- many mods
; Updated: 08-21-00  Bob Kehoe -- use new var struct, add some variables (eg. ivalue calc.)
; Updated: 12-07-00  Bob Kehoe -- added duration parameter to var struct
; Updated: 11-20-16  Govinda Dhungana -- added objid optional argument for printing photometry
; Updated: 29-07-20  Chloe Lawrence -- added printf to filtered observations for individual objects
; Updated: 29-07-20  Caroline Kuczek -- added printout of coordinates for selected objects
;******************************************************************************


if N_params() lt 3 then begin
   print, 'Syntax var = find_burst_cl(match,mindeltaold,minsig,minchisq=minchisq,emask=emask,rmask=rmask,/fakes,rotse=experiment,log=log,refra=refra,refdec=refdec,objid=objid,radius=radius)'
  return, -1
endif

; Initialization

nobj = long((size(oldmatch.m))[2])
totobs = (size(oldmatch.m))[1]
print, 'Total number of objects found in ', totobs, ' epochs = ', nobj
match = oldmatch
bad_eflags = ['ATEDGE', 'SATURATED', 'APINCOMPL']
bad_rflags = ['NOISYPIX', 'HOTPIX', 'STRANGEPIX', 'BADPOS', 'NOTEMPL', 'PHOTSDEV']
if not keyword_set(radius) then radius=0.001 ;;- radius from given ra, dec 
if not keyword_set(emask) then emask = set_flags(bad_eflags, type='EFLAGS')
if not keyword_set(rmask) then rmask = set_flags(bad_rflags, type='RFLAGS')

;; Filter observations
if keyword_set(objid) then begin
   goodobj=objid ;- single specified object

   nobs=n_elements(match.m[*,objid]) ;; - no. of observation for that objid
   print, "Photometry completed for this object ID", objid 
   filename = log + '.dat'
   GET_LUN, lunitno
   OPENW, lunitno, filename
   for ii=0,nobs-1 do begin
      print, match.jd[ii], match.m[ii,objid],match.merr[ii,objid] ;; This is probably enough.
      printf, lunitno, match.jd[ii], match.m[ii,objid],match.merr[ii,objid]
   endfor
   FREE_LUN, lunitno
   print, "Printing result: " + filename

endif else begin

   if (n_elements(refra) gt 0) and (n_elements(refdec) gt 0) then begin
      goodobj = where((abs(match.ra-refra) lt radius) and (abs(match.dec-refdec) lt radius), nmatch)
      nobs=n_elements(match.jd)
      
      ; No coordinate match
      if nmatch eq 0 then begin
         print, 'No object found within the search radius.'
         return, -1
      endif

      ; Exactly one coordinate match
      if nmatch eq 1 then begin
         iobj = goodobj[0]
	 print, iobj
         filename = log + '.dat'

      ;; print the lightcurve if there is only one object
	 GET_LUN, lunitno
	 OPENW, lunitno, filename
         for ii = 0,nobs-1 do begin
            print, match.jd[ii], match.m[ii,iobj],match.merr[ii,iobj],format='(%" %f     %f     %f")'
            printf, lunitno, match.jd[ii], match.m[ii,iobj],match.merr[ii,iobj],format='(%" %f     %f     %f")'
         endfor
         FREE_LUN, lunitno
	 print, "Printing result: " + filename
      endif

   endif else begin

      if keyword_set(minchisq) then begin
         goodobj = filter_obs(match,mindelta,minsig,chisq=minchisq,emask=emask,rmask=rmask)
      endif else begin
         goodobj = filter_obs(match,mindelta,minsig,emask=emask,rmask=rmask)
   endelse
;Printout of coordinates for selected objects
      get_lun, lunitno
      print, lunitno
      if keyword_set(log) then begin
         txtfile = log + '.txt'
         openw,lunitno,txtfile
         if n_elements(goodobj) gt 0 then begin
            nobjs=n_elements(goodobj)
            for j=0,nobjs-1 do begin
                print, match.ra[goodobj[j]],match.dec[goodobj[j]],goodobj[j]
                printf,lunitno,match.ra[goodobj[j]],match.dec[goodobj[j]],goodobj[j]
            endfor
         endif
         free_lun,lunitno
	 print, "Printing result: " + txtfile
      endif
   endelse
endelse

 if keyword_set(experiment) then begin
   rotse = experiment
endif else begin
   rotse = 1
endelse
print, 'rotse ',rotse

;nvar = long((size(goodobj))[1])
nvar=long(size(goodobj,/N_ELEMENTS))
;;print, "goodobj", goodobj
print, 'Number of variables found = ',nvar

; Calculate lightcurve quantities from good observations.

;var = make_var_struct(nvar, totobs)
;var.ptr = long(goodobj)

if (nvar gt 0L) then begin
   var = make_var_struct(nvar, totobs)
   for k = 0L,nvar-1L do begin

;     Get identification for this variable

      var[k].ptr = long(goodobj[k])
      print, var[k].ptr
      ptr = var[k].ptr
      var[k].name = make_rotse_name(match.ra[ptr], match.dec[ptr])

;     Find observations where source was not seen but should have been

      count = 0
      if rotse eq 1 then begin
         misses = where(match.m[*,ptr] eq -1.0 and var[k].avgmag lt (match.stat.m_lim-1.0), count)
      endif
      var[k].nmiss = count
      if (var[k].nmiss gt 0) then var[k].obs[misses].state = 1

;     Find observations where source seen, and calculate position and duration information

      okobs = where(match.flags[*,ptr] gt -1, count)
      var[k].nobs = count
      print, count
      if (count gt 0) then begin
         var[k].obs[okobs].state = 2
         var[k].duration = max(match.jd[okobs]) - min(match.jd[okobs])
      endif
      dra = conv2deg(match.dra[*,ptr])
      ddec = conv2deg(match.ddec[*,ptr])
      var[k].obs.dis = sqrt(dra^2.0 + ddec^2.0)
      var[k].obs.posangle = atan(ddec/dra)
      var[k].pos_sdv = (stddev(dra)+stddev(ddec)) / 2.0
      var[k].posrange = (max(dra)-min(dra)) > (max(ddec)-min(ddec))
;      var[k].duration = max(match.jd[okobs]) - min(match.jd[okobs])

;     Obtain list of good observations and use to calculate lightcurve information

      gdobs = where(match.flags[*,ptr] gt -1 and $
		check_flags(emask,match.flags[*,ptr],type='EFLAGS') eq 0 and $
		check_flags(rmask,match.rflags[*,ptr],type='RFLAGS') eq 0, count)
      var[k].ngdobs = count
      if (count gt 1) then begin
         var[k].obs[gdobs].state = 3
         var[k].obs[gdobs].err = sqrt(match.merr[gdobs,ptr]^2.0 + $
				     (match.msys[gdobs,ptr]/200.0)^2.0)
         kth_var = var[k]
         if rotse eq 1 then begin
            lightcurve, match.m[gdobs,ptr], var[k].obs[gdobs].err, match.stat[gdobs].m_lim,$
		  mindelta, minsig, kth_var
         endif
         var[k] = kth_var
         ivalue,match.m[gdobs,ptr],var[k].obs[gdobs].err,ival,mn_iter=0
         var[k].ival = ival
         ivalue,match.m[gdobs,ptr],var[k].obs[gdobs].err,ival,mn_iter=4,/robust
         var[k].ival2 = ival
      endif
   endfor
endif
   
if keyword_set(log) then begin
   txtfile = log + '.txt'
   ;; print if var.ptr more than 1
;   if n_elements(var.ptr) gt 1 then begin
;      print_var,match,var,fname=txtfile
;   endif else begin ;; simply write a light curve only
;      lun=5
;      openw,lun,txtfile
;      nobs=n_elements(match.m[*,goodobj])
;      for ii=0,nobs-1 do begin
;         printf,lun,match.jd[ii], match.m[ii,goodobj],match.merr[ii,goodobj],format='(%" %f     %f     %f")'
;      endfor
;      close,lun
;   endelse
   set_plot, 'ps'
   !P.MULTI = [0,2,3]
   psfile = log + '.ps'
   device, file=psfile
   if keyword_set(experiment) then begin
      lcplot2,match,indgen(totobs),var.ptr,/good,/syserr,/offset,emask=emask,rmask=rmask,/nowait,rotse=rotse
   endif else begin
      lcplot2,match,indgen(totobs),var.ptr,/good,/syserr,/offset,emask=emask,rmask=rmask,/nowait
   endelse
   device, /close
   set_plot, 'X'
endif

   return, var
end
