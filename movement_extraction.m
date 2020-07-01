
%% select pathway
clear; clc; close all; cnt = 0;

for i = 1:1 % 몇개할건지 적음.
    cnt = cnt +1;
    try load .dir.mat; catch; dir_nm = [cd(), filesep];  end     
    [file_nm, dir_nm] = uigetfile(fullfile(dir_nm, '*.avi'));
    filepath = [dir_nm, file_nm];

    path_save{cnt,1} = filepath;
end
%% select folders
clear; clc; close all; cnt = 0; extension = '.avi'
for i = 1:4% 몇개할건지 적음.
    
    try load .dir.mat; catch; dir_nm = [cd(), filesep];  end     
    [file_nm, dir_nm] = uigetfile(fullfile(dir_nm, '*.avi'));
    filepath = [dir_nm, file_nm];
    
    aviFileList = msCamVideoFileDetection(dir_nm, extension);
    
    for j = 1:size(aviFileList,2)
        cnt = cnt +1;
    	path_save{cnt,1} = cell2mat(aviFileList(j));
    end
end

%%

for cnt = 1:size(path_save,1)
    clear msdiff
    filepath = cell2mat(path_save(cnt,1))
    v = VideoReader(filepath);
    for frame = 1: v.NumberOfFrames
        disp([num2str(frame) ' / ' num2str(v.NumberOfFrames)])
        tmpFrame = double(v.read(frame));
        msFrame = tmpFrame(:,:,1);

        if frame > 1
            tmp = abs(msFrame-preFrame);
            msdiff(frame) =  mean(tmp(:));
        end

        preFrame = msFrame;
    end

    gaussFilter = gausswin(30); 
    gaussFilter = gaussFilter / sum(gaussFilter); % Normalize.

    msdiff_gauss = conv(msdiff,gaussFilter, 'valid');
    
    msmatrix = msdiff_gauss;
    msmax = max(msmatrix); msmin = min(msmatrix); diff = (msmax - msmin)/10;
    tmpmax = -Inf; savemax = NaN;
    
    for j = 0:9
        c1 = (msmatrix >= (msmin + diff * j));
        c2 = (msmatrix < (msmin + diff * (j+1)));

        if tmpmax < sum(c1 .* c2)
            tmpmax = sum(c1 .* c2); savemax = j;
        end
    end

    c1 = (msmatrix >= (msmin + diff * savemax));
    c2 = (msmatrix < (msmin + diff * (savemax+1)));
    

    ix = c1 .* c2;
    tmpsum = 0; tmpcnt = 0;
    for i = 1:size(ix, 2)
        if ix(i) == 1
            tmpsum = tmpsum + msmatrix(1, i);
            tmpcnt = tmpcnt + 1;
        end
    end
    
    mscut = tmpsum / tmpcnt;
    thr = mscut + 0.15; % this threshold will be modified by group blinded experimenter
    
    aline = ones(1,size(ix, 2));
    aline = aline .* thr;
    
    figure(1)
    plot(msdiff_gauss)
    hold on
    plot(aline)
    hold off

    saveas(1,[filepath '.png'])
    save([filepath '.mat'],'msdiff_gauss')
end


sum(msdiff_gauss > 2.5)
























