function sequence=processtextfile(fname,outfname);
%converts textfile into a sequence of numbers to be used in RNN experiments
%include argument outfname to save "sequence" to a .mat file, which will allow "sequence" to be
%loaded more quickly with the command load(outfname)


fid = fopen(fname);
bytes=fread(fid,'*uint8');
u = unique(bytes);
sequence = zeros(length(bytes),1,'single');
for i=1:length(u)
    f = logical(bytes==u(i));
    sequence(f) = i;
end

if exist('outfname','var')
    
save('outfname','sequence')
display('saved')
end
end