pro print_var, match,var,fname=fname

; Created:  05-31-00  Bob Kehoe
; Updated:  11-29-16 Govinda Dhungana
   ;;       : fix index on var, e.g var.ptr[k] --> var[k].ptr
   ;;       : fix var.dis --> var.obs.dis

if N_params() lt 2 then begin
   print, 'Syntax print_var,match,var,fname=fname'
   return
endif

num = (size(var.ptr))[1]

if keyword_set(fname) then begin
   lun = 5
   openw,lun,fname
   for k = 0,num-1 do begin
      h = where(match.m[*,var[k].ptr] gt -1.0)
      printf, lun, 'K =', var[k].ptr, ':  RA=', match.ra[var[k].ptr], $
		' DEC=', match.dec[var[k].ptr]
      printf, lun, '     avgmag, delta = ',var[k].avgmag,var[k].delta
      printf, lun, '     mag = ',match.m[h,var[k].ptr]
      printf, lun, '     dis = ',var[k].obs[h].dis
      printf, lun, '     eflags = ',match.flags[h,var[k].ptr]
      printf, lun, '     msys = ',match.msys[h,var[k].ptr]
      printf, lun, '     rflags = ',match.rflags[h,var[k].ptr]
   endfor
   printf, lun, ' '
   printf, lun, ' '
   printf, lun, ' '
   for k = 0,num-1 do printf, lun, var[k].name, var[k].maxdelta, $
			var[k].maxerr, var[k].chisqcl
   printf, lun, ' '
   printf, lun, ' '
   printf, lun, ' '
   for k = 0,num-1 do begin
	printf, lun, var[k].name, var[k].delta, var[k].avgmag, $
			match.ra[var[k].ptr], match.dec[var[k].ptr]
   endfor
   close,lun
endif else begin

   for k = 0,num-1  do begin
      h = where(match.m[*,var[k].ptr] gt -1.0)
      print, 'K =',var[k].ptr, ':  RA=',match.ra[var[k].ptr],' DEC=',$
		match.dec[var[k].ptr]
      print, '     avgmag, delta = ',var[k].avgmag,var[k].delta
      print, '     mag = ',match.m[h,var[k].ptr]
      print, '     dis = ',var[k].obs[h].dis 
      print, '     eflags = ',match.flags[h,var[k].ptr]
      print, '     msys = ',match.msys[h,var[k].ptr]
      print, '     rflags = ',match.rflags[h,var[k].ptr]
   endfor
   print, ' '
   print, ' '
   print, ' '
   for k = 0,num-1 do print, var[k].name, var[k].maxdelta, var[k].maxerr, $ 
			 var[k].chisqcl 
   print, ' '
   print, ' '
   print, ' '
   for k = 0,num-1 do begin
	print, var[k].name, var[k].delta, var[k].avgmag, $
			match.ra[var[k].ptr], match.dec[var[k].ptr]
   endfor
endelse

end


