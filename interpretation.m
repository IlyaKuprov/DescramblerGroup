% Interpretation process for a simple two-layer DEERNet, as
% presented in our paper on the subject:
%
%          https://doi.org/10.1073/pnas.2016917118
%
% Calculation time: minutes on a Tesla V100 GPU.
%
% i.kuprov@soton.ac.uk

function interpretation()

% Load the net
load('deernet_256_80.mat','checkpoint');
W=checkpoint.net.IW{1};
U=checkpoint.net.LW{2,1};

% Make a training database
parameters=library_dd();
parameters.ntraces=1e6;
parameters.np_time=256;
parameters.np_dist=256;
parameters.expt='deer';
library=deer_lib_gen([],parameters);

% Move to double precision
inputs=double(library.deer_noisy_lib);
 
% Descramble first layer
P1=descramble(W*inputs,1e4,zeros(80));

% Descramble the second layer
mult=norm(tansig(W*inputs),2)/norm(U',2);
P2=descramble([mult*U' tansig(W*inputs)],1e4,zeros(80));

%% Figure 2

% Plot the original
figure(); subplot(1,4,1); imagesc(W'); 
title('raw weight m.'); axis tight;
set(gca,'YDir','normal');
set(gca,'xtick',[20 40 60],'FontSize',10);
ylabel('input dimension, points');
xlabel('link dim, pts');
text(62,240,'$$\mathbf{W}^{\top}_1$$',...
            'Interpreter','latex','Color','white');

% Descramble
W1=P1*W;
subplot(1,4,2); imagesc(W1'); 
title('descrambled'); axis tight;
set(gca,'YDir','normal');
set(gca,'xtick',[20 40 60],'FontSize',10); 
set(gca,'ytick',[]);
xlabel('link dim, pts');
text(62,240,'$$\mathbf{W}^{\top}_1$$',...
            'Interpreter','latex','Color','white');

% Fourier transform and symm
W2=fftshift(fft2(W1));
W2=abs(W2)+abs(circshift(fliplr(W2),1,2));
subplot(1,4,3); imagesc(abs(W2')); axis tight;
set(gca,'YDir','normal');
title('descr. + fft');
set(gca,'xtick',[20 40 60],'FontSize',10);  
set(gca,'ytick',[]);
xlabel('link dim, pts');
text(32,240,'$$(\mathbf{F}_{+}\mathbf{W}_{1}\mathbf{F}_{-})^{\top}$$',...
            'Interpreter','latex','Color','white');

% Annotate FT
line([5 75],[162 162],'Color','white');
line([5 75],[95 95],'Color','white');
text(5,172,'bandpass','Color','white');
line([15 35],[129 129],'Color','white');
line([15 15],[70 129],'Color','white');
text(12,65,'notch','Color','white');

% Cubic curve
subplot(1,4,4);
imagesc(W2'); hold on; axis tight;
set(gca,'YDir','normal');
title('descr. + fft');
set(gca,'xtick',[25 30 35 40 45 50 55],'FontSize',10);  
set(gca,'ytick',[]);
xlabel('link dim, pts');
x=linspace(1,80,1000);
y=+0.05*(x-41).^3+129;
plot(x,y,'Color','white');
y=-0.05*(x-41).^3+129;
plot(x,y,'Color','white');
xlim([1 80]); ylim([1 256]); zoom(3);
text(35.5,90,'$$y= \pm x^3$$','Interpreter','latex',...
             'Color','white','FontSize',12);

scale_figure([1.5,0.8])

%% Figure 3A

% Plot the original
figure();
subplot(2,1,1); imagesc(U'); 
title('raw weight matrix'); axis equal; axis tight;
set(gca,'YDir','normal');
set(gca,'ytick',[20 40 60]);
ylabel('link dim, pts');
xlabel('output dimension, points');

% Descramble
U1=U*P2';
subplot(2,1,2); 
imagesc(U1'-mean(U1',1)); %#ok<UDIM>
title('descrambled weight matrix'); 
axis equal; axis tight;
set(gca,'YDir','normal');
set(gca,'ytick',[20 40 60]); 
xlabel('output dimension, points');
ylabel('link dim, pts');

% Annotate plot
line([15 15],[1 80],'Color','white');
line([242 242],[1 80],'Color','white');
text(18,5,'fast','Color','white','Rotation',90);
text(237,5,'slow','Color','white','Rotation',90);


%% Figure 3B
figure()
subplot(2,2,1); imagesc(U1'*U1);
axis square;
xlabel('link dim, pts');
ylabel('link dim, pts');
text(52,8,'$$\mathbf{W}^{\top}_2 \mathbf{W}_2$$',...
          'Interpreter','latex','Color','white');
subplot(2,2,3); imagesc(U1*U1');
axis square;
xlabel('output dim, pts');
ylabel('output dim, pts');
text(170,25,'$$\mathbf{W}_2 \mathbf{W}^{\top}_2$$',...
            'Interpreter','latex','Color','white');

% Singular value decomposition
[L,~,R]=svd(U1,'econ');

% Sinusoids
subplot(2,2,2); plot(R(:,9:13)+[-2 -1 0 1 2]/2);
axis tight; grid on;
title('singular vectors, link dim');
set(gca,'ytick',[]); 
xlabel('link dimension, pts')
ylabel('amplitude, a.u.');

% Chebyshev fits
L=L(:,1:5)-mean(L(:,1:5));
subplot(2,2,4); hold on;
C=chebyshevT(0,linspace(-1,1,256))'; a=L(:,1)'/C';
plot(L(:,1)+0.5,'.','Color',[0.47 0.67 0.19]);
plot(a*C+0.5,'-','Color',[0.47 0.67 0.19]);
C=chebyshevT(1,linspace(-1,1,256))'; a=L(:,2)'/C';
plot(L(:,2)+0.35,'.','Color',[0.49 0.18 0.56]);
plot(a*C+0.35,'-','Color',[0.49 0.18 0.56]);
C=chebyshevT(2,linspace(-1,1,256))'; a=L(:,3)'/C';
plot(L(:,3)+0,'.','Color',[0.93 0.69 0.13]);
plot(a*C+0,'-','Color',[0.93 0.69 0.13]);
C=chebyshevT(3,linspace(-1,1,256))'; a=L(:,4)'/C';
plot(L(:,4)-0.25,'.','Color',[0.85 0.33 0.10]);
plot(a*C-0.25,'-','Color',[0.85 0.33 0.10]);
C=chebyshevT(4,linspace(-1,1,256))'; a=L(:,5)'/C';
plot(L(:,5)-0.5,'.','Color',[0.00 0.45 0.74]);
plot(a*C-0.5,'-','Color',[0.00 0.45 0.74]);
axis tight; grid on; box on;
title('singular vectors, output dim');
set(gca,'ytick',[]); 
xlabel('output dimension, pts')
ylabel('amplitude, a.u.');

end

