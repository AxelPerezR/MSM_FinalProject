% PCA: Signal Compression
% 22/12/23

% Axel Perez & Abednego Sauri

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1. Visualizing our data
disp('-----------------------------------------------------------------------------------');
disp('1. Visualizing our data');
file = 'speech_8kHz.wav';
[y,Fs]=audioread(file);

% This line is to hear the audiofile
%playblocking(audioplayer(y,Fs)); 

%Plooting the sound wave
fig1 = figure(1);
ax1 = nexttile;

hold(ax1,'on');

t=linspace(0,length(y)/Fs,length(y));

title('Amplitude per time of our original data');
plot(t,y);
xlabel('Time (s)');
ylabel('Amplitude');
ylim([-1,1]);
grid on;

hold(ax1,'off');

% Visualazing standarize data

ys = standarize(y);

fig2 = figure(2);
ax2 = nexttile;

hold(ax2,'on');
t=linspace(0,length(y)/Fs,length(y));
title('Original data vs data after standardization');
plot(t,ys);
plot(t,y);
xlabel('Time (s)');
ylabel('Amplitude');
ylim([-1,1]);
legend('Standardized data', 'Original data');
grid on;
hold(ax2,'off');

%playblocking(audioplayer(ys,Fs))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2. Minimum number of principal components to understand the message.
disp('-----------------------------------------------------------------------------------');
disp('2. We write different audiofiles to be analyzed.');
% 20%
[Fs20,X20,E20,yd20,Xd20,Ed20] = encoding_decoding(file,100,20);
%audiowrite('audio20.wav',yd20,Fs20);
% 30%
[Fs30,X30,E30,yd30,Xd30,Ed30] = encoding_decoding(file,100,30);
%audiowrite('audio30.wav',yd30,Fs30);
% 50%
[Fs50,X50,E50,yd50,Xd50,Ed50] = encoding_decoding(file,100,50);
%audiowrite('audio50.wav',yd50,Fs50);
% 70%
[Fs70,X70,E70,yd70,Xd70,Ed70] = encoding_decoding(file,100,70);
%audiowrite('audio70.wav',yd70,Fs70);
% 90%
[Fs90,X90,E90,yd90,Xd90,Ed90] = encoding_decoding(file,100,90);
%audiowrite('audio90.wav',yd90,Fs90);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 3. Fixed space saved

disp('-----------------------------------------------------------------------------------');
disp('3. For a fixed space saved we compute the distortion values for different groupings');

space_saved = [90,70,50,30,10];
grouping = [10,50,100,150,190];

for i=1:length(space_saved)
    disp(['Space saving: ', num2str(space_saved(i)),'%']);
    for j=1:length(grouping)
        k = grouping(j)/100*(100-space_saved(i));
        [Fs,X,E,yd,Xd,Ed] = encoding_decoding(file,grouping(j),k);
        Dis = distortion(y,yd);
        %Storing all values
        dpfss(i,j) = Dis; 
        disp(['D(g=', num2str(grouping(j)),'): ', num2str(Dis)]);
    end
    %Computing the mean value
    mean_d(i)=mean(dpfss(i,:));
end

% Plotting
% Graph of distortion values

fig3 = figure(3);
ax3 = nexttile;
hold(ax3,'on');

title('Distortion for different groupings and fixed space saving');
xlabel('Grouping');
ylabel('Distortion');
grid(ax3,'on');

for i=1:length(space_saved)
    scatter(grouping, dpfss(i,:), 'filled');
    yline(mean_d(i), '--', mean_d(i));
end

ylim([0,max(dpfss(:))+0.1*max(dpfss(:))]);

hold(ax3,'off');

%Graph of mean values
%Simple linear regression

disp('3.1. We approximate the relation between mean distortion using a linear approximation');
b=mean_d/space_saved;
lr=b*space_saved;
disp(['Linear regression equation: Mean-distortion=',num2str(b),'*Space-saved']);
R2 = 1 - sum((mean_d - lr).^2)/sum((lr - mean(lr)).^2);
disp(['Linear regression constant: R^2=', num2str(R2)]);

fig4 = figure(4);
ax4 = nexttile;
hold(ax4,'on');

title('Mean value of the distortion for different space savings');
xlim([0 100]);
xlabel('Space saving (%)');
ylabel('Mean distortion');
grid(ax4,'on');

%Plotting points
scatter(space_saved,mean_d, 'filled');
%Plotting linear regression
plot(space_saved,lr);
legend('Distorion values', 'Fitting curve', 'Location', 'southeast');
hold(ax4,'off');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 4. Comparing different formulas to compute the distortion.

disp('-----------------------------------------------------------------------------------');
disp('For a fixed space saved of 10% we compute different D values.');

for i=1:length(grouping)
    disp(['For a grouping of ', num2str(grouping(i))]);
    [Fs,X,E,yd,Xd,Ed] = encoding_decoding(file,grouping(i),grouping(i)/100*90);
    Dis = distortion(y,yd);
    disp(['D = ', num2str(Dis)]);
    DisI = distortionI(X,E,Xd,Ed,y,grouping(i));
    disp(['D.I = ', num2str(DisI)]);
    error(i) = abs(100-DisI*100/Dis);
    disp(['Error(%) = ', num2str(error(i))]);
end
disp(['Error Mean value (%) = ', num2str(mean(error))]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 5. Comparing compresed signal
disp('-----------------------------------------------------------------------------------');
disp('5. Comparing compressed signal');
% For a space saving of 50% and 10%

[Fs50,X50,E50,yd50,Xd50,Ed50] = encoding_decoding(file,100,100/100*50);
[Fs10,X10,E10,yd10,Xd10,Ed10] = encoding_decoding(file,100,100/100*10);

%Plooting the sound wave
fig5 = figure(5);
ax5 = nexttile;

hold(ax5,'on');

t=linspace(0,length(y)/Fs,length(y));
td50=linspace(0,length(yd50)/Fs50,length(yd50));
td10=linspace(0,length(yd10)/Fs10,length(yd10));
title('Sound wave for different space saved');
plot(t,y);
plot(td50,yd50);
plot(td10,yd10);
xlabel('Time (s)');
ylabel('Amplitude');
ylim([-1,1]);
legend('Original', '50% space saved', '10% space saved');
grid on;

hold(ax5,'off');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 6. Analysing different audio signals
disp('-----------------------------------------------------------------------------------');
disp('6. Analysing different audio signals');

file1 = 'bubbles.wav';
[y1,Fs1]=audioread(file1);

file2 = 'tada.wav';
[y2,Fs2]=audioread(file2);

% Comparing compressed signals

% File 1
[Fs150,X150,E150,yd150,Xd150,Ed150] = encoding_decoding(file1,100,100/100*50);
[Fs110,X110,E110,yd110,Xd110,Ed110] = encoding_decoding(file1,100,100/100*10);

% File 2
[Fs250,X250,E250,yd250,Xd250,Ed250] = encoding_decoding(file2,100,100/100*50);
[Fs210,X210,E210,yd210,Xd210,Ed210] = encoding_decoding(file2,100,100/100*10);

% Distortion values
space_saved = [90,70,50,30,10];
grouping = [10,50,100,150,190];

for i=1:length(space_saved)
    for j=1:length(grouping)
        k = grouping(j)/100*(100-space_saved(i));
        [Fs1,X1,E1,yd1,Xd1,Ed1] = encoding_decoding(file1,grouping(j),k);
        Dis1 = distortion(y1,yd1);
        dpfss1(i,j) = Dis1;
    end
    mean_d1(i)=mean(dpfss1(i,:));
end

for i=1:length(space_saved)
    for j=1:length(grouping)
        k = grouping(j)/100*(100-space_saved(i));
        [Fs2,X2,E2,yd2,Xd2,Ed2] = encoding_decoding(file2,grouping(j),k);
        Dis2 = distortion(y2,yd2);
        dpfss2(i,j) = Dis2;
    end
    mean_d2(i)=mean(dpfss2(i,:));
end

% Plotting
% Graph of distortion values

fig6 = figure(6);
ax6 = nexttile;
hold(ax6,'on');

title('Distortion for different groupings and fixed space saving for bubbles.wav');
xlabel('Grouping');
ylabel('Distortion');
grid(ax6,'on');

for i=1:length(space_saved)
    scatter(grouping, dpfss1(i,:), 'filled');
    yline(mean_d1(i), '--', mean_d1(i));
end

ylim([0,max(dpfss1(:))+0.1*max(dpfss1(:))]);

hold(ax6,'off');

fig7 = figure(7);
ax7 = nexttile;
hold(ax7,'on');

title('Distortion for different groupings and fixed space saving for tada.wav');
xlabel('Grouping');
ylabel('Distortion');
grid(ax7,'on');

for i=1:length(space_saved)
    scatter(grouping, dpfss2(i,:), 'filled');
    yline(mean_d2(i), '--', mean_d2(i));
end

ylim([0,max(dpfss2(:))+0.1*max(dpfss2(:))]);

hold(ax7,'off');


%Plooting the sound wave
fig8 = figure(8);
ax8 = nexttile;

hold(ax8,'on');

t1=linspace(0,length(y1)/Fs1,length(y1));
td150=linspace(0,length(yd150)/Fs150,length(yd150));
td110=linspace(0,length(yd110)/Fs110,length(yd110));
title('bubbles.wav compression for different space saved');
plot(t1,y1);
plot(td150,yd150);
plot(td110,yd110);
xlabel('Time (s)');
ylabel('Amplitude');
ylim([-1,1]);
xlim([0,max(t1)]);
legend('Original', '50% space saved', '10% space saved');
grid on;

hold(ax8,'off');

fig9 = figure(9);
ax9 = nexttile;

hold(ax9,'on');

t2=linspace(0,length(y2)/Fs2,length(y2));
td250=linspace(0,length(yd250)/Fs250,length(yd250));
td210=linspace(0,length(yd210)/Fs210,length(yd210));
title('tada.wav compression for different space saved');
plot(t2,y2);
plot(td250,yd250);
plot(td210,yd210);
xlabel('Time (s)');
ylabel('Amplitude');
ylim([-1,1]);
xlim([0,max(t2)]);
legend('Original', '50% space saved', '10% space saved');
grid on;

hold(ax9,'off');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Functions

% Function to encode and decode the audiofile
function [Fs,X,E,yd,Xd,Ed] = encoding_decoding(file,g,k)

    % 1. Read our data
    [y,Fs]=audioread(file);

    % 2. Transform our data matrix

    % 2.1. Create and reshape X
    n0 = ceil(size(y,1)/g)*g-size(y,1);

    yre = [y;zeros(n0,1)];

    X = reshape(yre,length(yre)/g,[]);

    % 3. Standardized our data

    % 3.1. Compute X centered (Xc)
    Xc = X - mean(X);

    % 3.2. Compute X standarized (Xs)
    stdX = std(X);
    for i=1:length(stdX)
        if stdX(i) == 0
           %M=sort(stdX);
           stdX(i) = 0.00001;
        end
    end
    Xstd = ones(length(yre)/g,1)*stdX;
    Xs = Xc./Xstd;


    % 4. Compute the Covariance Matrix
    M = (Xs'*Xs)/(g-1);

    % 5. Eigenvalue Decomposition
    [E,D] = eig(M);

    % 6. Sort Eigenvalues and Eigenvectors
    D = fliplr(D);
    D = flipud(D);
    E = fliplr(E);
    E = flipud(E);

    % 7. Select Principal Components
    % k is already given

    % 8. Encode Data (Xe)

    Xe = Xs* E(:, 1:k);

    % 9. Decode data (Xd)

    Xd = Xe * E(:, 1:k)';

    %9.1. Compute the Covariance Matrix of decoded data
    Md = (Xd'*Xd)/(g-1);
    TF = isnan(Md);
    Md(TF) = 0;
        
    %9.2. Eigenvalue Decomposition
    [Ed,Dd] = eig(Md);

    %9.3. Sort Eigenvalues and Eigenvectors
    Dd = fliplr(Dd);
    Dd = flipud(Dd);
    Ed = fliplr(Ed);
    Ed = flipud(Ed);

    % 10. Inverse standardization.

    Xd = Xd.*Xstd + mean(X);

    % 11. Results

    yd = reshape(Xd,[],1);
end

% Function to standarize the data
function ys = standarize(y)
    yc = y - mean(y);
    ystd = std(y);
    ys = yc./ystd;
end

% Function to compute the distortion with the given formula
function Dis = distortion(y,yd)
    Sumd = 0;
    for i=1:length(y)
       Sumd = Sumd + (y(i,1)-yd(i,1))^2;
    end
    Dis = Sumd/length(y);
end

% Function to compute the distortion using the formula that use the
% information
function DisI = distortionI(X,E,Xd,Ed,y,g)
    %Total information
    P = X*E;
    p = 0;
    %Transmitted information
    Pd = Xd*Ed;
    pd = 0;

    for i=1:length(y)/g
        for j=1:g
            p = p + P(i,j)^2;
            pd = pd + Pd(i,j)^2;
        end
    end
    Itot = p/(i-1);
    I = pd/(i-1);
    DisI = (Itot-I)/g;
end
