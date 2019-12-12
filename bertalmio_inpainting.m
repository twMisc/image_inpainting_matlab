clear;
%camera man example
%mask = double(rgb2gray(imread('cameraman3_mask.png')));
%img=double(rgb2gray(imread('cameraman3.png')));

% Lin's example
img=double(rgb2gray(imread('example.jpg')));

% dummy example 1
%img=[128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 ;128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 ;128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 ;128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 ;128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 ;128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 ;128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 ;128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 ;128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 ;128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 ; 128 128 128 128 128 128 128 128 128 128 255 255 255 255 255 128 128 128 128 128 128 128 128 128 128; 128 128 128 128 128 128 128 128 128 128 255 255 255 255 255 128 128 128 128 128 128 128 128 128 128; 128 128 128 128 128 128 128 128 128 128 255 255 255 255 255 128 128 128 128 128 128 128 128 128 128; 0 0 0 0 0 0 0 0 0 0 255 255 255 255 255 0 0 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0 0 0 255 255 255 255 255 0 0 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0];

% dummy example 2
%load('dummly.mat');

% maxiter : number of the whole loop
% maxiter1 : number of in inpainting loop
% maxiter2 : number of p.-m. model diffusion
maxiter = 500;
maxiter1=15;
maxiter2=2;

%imshow(uint8(img));

%define mask if "white enough"
mask=zeros(size(img));
mask(img>240)=255; %mask defined all black and inpainting region white

% copy of the original image
oimg=img;

% paramaeters for p.-m. diffusion
nu=1;
[N, M]=size(img);
dx=1/N;
dy=1/M;
dt=0.1*dx^2/nu;
lambda1=dt*nu/(dx)^2;
lambda2=dt*nu/(dy)^2;
x=dx*(1/2:1:N-1/2);
y=dy*(1/2:1:M-1/2);
C=zeros(N+2,M+2);
%coef=@(x) nu;
%coef=@(x) 1./sqrt(1+x);
K=3;
%coef=@(x) 1./(1+x./K);
coef=@(x)  exp(-x./K);

% dt for the inpainting alg.
delta_t = 0.002;

for itr = 1:maxiter
    for k=1:maxiter1
    % bound=edge(mask,'log'); % detect boundary
    img_new=img;
        for x=3:(size(img,1)-2)
            for y=3:(size(img,2)-2)
                if (mask(x,y)==255) % if it belongs to the mask
                    %mask(x-1,y)=0;mask(x+1,y)=0;mask(x,y-1)=0;mask(x,y+1)=0;mask(x,y)=0;
                    %3.4
                    Ix=(img(x+1,y)-img(x-1,y))/2;
                    Iy=(img(x,y+1)-img(x,y-1))/2;
                    denN=sqrt((Ix'*Ix)+(Iy'*Iy)+0.000000001); %epsilon 0.000001 to aviod divide by zero
                    Nx=Ix/denN;
                    Ny=Iy/denN;
                    %3.3
                    p1=x+1;q1=y;
                    Lnew1=img(p1+1,q1)+img(p1-1,q1)+img(p1,q1+1)+img(p1,q1-1)-4*img(p1,q1);
                    p2=x-1;q2=y;
                    Lnew2=img(p2+1,q2)+img(p2-1,q2)+img(p2,q2+1)+img(p2,q2-1)-4*img(p2,q2);
                    p3=x;q3=y+1;
                    Lnew3=img(p3+1,q3)+img(p3-1,q3)+img(p3,q3+1)+img(p3,q3-1)-4*img(p3,q3);
                    p4=x;q4=y-1;
                    Lnew4=img(p4+1,q4)+img(p4-1,q4)+img(p4,q4+1)+img(p4,q4-1)-4*img(p4,q4);
                    deLx=Lnew1-Lnew2;
                    deLy=Lnew3-Lnew4;
                    %3.6
                    beta=((deLx*(-Ny))+(deLy*Nx));
                    %3.5
                    Ixf=img(x+1,y)-img(x,y);
                    Ixb=img(x,y)-img(x-1,y);
                    Iyf=img(x,y+1)-img(x,y);
                    Iyb=img(x,y)-img(x,y-1);
                    Ixfm=min(Ixf,0);
                    IxfM=max(Ixf,0);
                    Ixbm=min(Ixb,0);
                    IxbM=max(Ixb,0);
                    Iyfm=min(Iyf,0);
                    IyfM=max(Iyf,0);
                    Iybm=min(Iyb,0);
                    IybM=max(Iyb,0);
                    if(beta>0)
                        ModeLI=sqrt((Ixbm*Ixbm)+(IxfM*IxfM)+(Iybm*Iybm)+(IyfM*IyfM));
                    elseif (beta<=0)
                        ModeLI=sqrt((IxbM*IxbM)+(Ixfm*Ixfm)+(IybM*IybM)+(Iyfm*Iyfm));
                    end
                    %3.2
                    It=beta*ModeLI;
                    %3.1 update
                    img_new(x,y)=img(x,y)+delta_t*(It);
                    
                    % set out of bound values back
                    if img_new(x,y)>255
                       img_new(x,y)=255;
                    end
                    if img_new(x,y)<0
                        img_new(x,y)=0;
                    end
                    %img_old(x,y)=double(uint8(int64(img_old(x,y)))); %another way for out of bound values 
                end
            end
        end
        img=img_new;
    end
    
    %copy img for the diffusion
    U=img;
    for k=1:maxiter2
        U_old=U;

         %update coefficient
         % update i:3:N, j:3:M
         Dx=(U(3:N,2:M-1)-U(1:N-2,2:M-1))./(2*dx);
         Dy=(U(2:N-1,3:M)-U(2:N-1,1:M-2))./(2*dy);
         C(3:N,3:M)=coef(Dx.^2+Dy.^2);

         % update i:2, j:3:M
         Dx=(U(2,2:M-1)-U(1,2:M-1))./(2*dx);
         Dy=(U(1,3:M)-U(1,1:M-2))./(2*dy);
         C(2,3:M)=coef(Dx.^2+Dy.^2);

         % update i:N+1, j:3:M
         Dx=(U(N,2:M-1)-U(N-1,2:M-1))./(2*dx);
         Dy=(U(N,3:M)-U(N,1:M-2))./(2*dy);
         C(N+1,3:M)=coef(Dx.^2+Dy.^2);

         % update i:3:N, j:2
         Dx=(U(3:N,1)-U(1:N-2,1))./(2*dx);
         Dy=(U(2:N-1,2)-U(2:N-1,1))./(2*dy);
         C(3:N,2)=coef(Dx.^2+Dy.^2);

         % update i:3:N, j:M+1
         Dx=(U(3:N,M)-U(1:N-2,M))./(2*dx);
         Dy=(U(2:N-1,M)-U(2:N-1,M-1))./(2*dy);
         C(3:N,M+1)=coef(Dx.^2+Dy.^2);

         % update i:2, j:2
         Dx=(U(2,1)-U(1,1))./(2*dx);
         Dy=(U(1,2)-U(1,1))./(2*dy);
         C(2,2)=coef(Dx.^2+Dy.^2);

         % update i:2, j:M+1
         Dx=(U(2,M)-U(1,M))./(2*dx);
         Dy=(U(1,M)-U(1,M-1))./(2*dy);
         C(2,M+1)=coef(Dx.^2+Dy.^2);

         % update i:N+1, j:M+1
         Dx=(U(N,M)-U(N-1,M))./(2*dx);
         Dy=(U(N,M)-U(N,M-1))./(2*dy);
         C(N+1,M+1)=coef(Dx.^2+Dy.^2);

         % update i:N+1, j:2
         Dx=(U(N,1)-U(N-1,1))./(2*dx);
         Dy=(U(N,2)-U(N,1))./(2*dy);
         C(N+1,2)=coef(Dx.^2+Dy.^2);

          % update i:1, j:3:M
         Dx=(U(1,2:M-1)-U(2,2:M-1))./(2*dx);
         Dy=(U(1,3:M)-U(1,1:M-2))./(2*dy);
         C(1,3:M)=coef(Dx.^2+Dy.^2);

          % update i:N+2, j:3:M
         Dx=(U(N-1,2:M-1)-U(N,2:M-1))./(2*dx);
         Dy=(U(N,3:M)-U(N,1:M-2))./(2*dy);
         C(N+2,3:M)=coef(Dx.^2+Dy.^2);

         % update i:3:N, j:1
         Dx=(U(3:N,1)-U(1:N-2,1))./(2*dx);
         Dy=(U(2:N-1,1)-U(2:N-1,2))./(2*dy);
         C(3:N,1)=coef(Dx.^2+Dy.^2);

         % update i:3:N, j:M+2
         Dx=(U(3:N,M)-U(1:N-2,M))./(2*dx);
         Dy=(U(2:N-1,M-1)-U(2:N-1,M))./(2*dy);
         C(3:N,M+2)=coef(Dx.^2+Dy.^2);

         U_old(2:N-1, 2:M-1)=U(2:N-1, 2:M-1) ...
          +0.5*lambda1*((C(4:N+1,3:M)+C(3:N,3:M)).*U(3:N,2:M-1)...
          -(2*C(3:N,3:M)+C(4:N+1,3:M)+C(2:N-1,3:M)).*U(2:N-1,2:M-1)...
                      +(C(2:N-1,3:M)+C(3:N,3:M)).*U(1:N-2,2:M-1)) ...
          +0.5*lambda2*((C(3:N,4:M+1)+C(3:N,3:M)).*U(2:N-1,3:M)...
          -(C(3:N,4:M+1)+2*C(3:N,3:M)+C(3:N,2:M-1)).*U(2:N-1,2:M-1)...
            +(C(3:N,2:M-1)+C(3:N,3:M)).*U(2:N-1,1:M-2));

        U_old(2:N-1, 2:M-1)=U_old(2:N-1, 2:M-1);

        % i=1 update
        U_old(1, 2:M-1)=U(1, 2:M-1) ...
          +0.5*lambda1*((C(3,3:M)+C(2,3:M)).*U(2,2:M-1)...
          -(2*C(2,3:M)+C(3,3:M)+C(1,3:M)).*U(1,2:M-1)...
                      +(C(1,3:M)+C(2,3:M)).*U(1,2:M-1)) ...
          +0.5*lambda2*((C(2,4:M+1)+C(2,3:M)).*U(1,3:M)...
          -(C(2,4:M+1)+2*C(2,3:M)+C(2,2:M-1)).*U(1,2:M-1)...
            +(C(2,2:M-1)+C(2,3:M)).*U(1,1:M-2));

        U_old(1, 2:M-1)=U_old(1, 2:M-1) ;

        % i=N update
         U_old(N, 2:M-1)=U(N, 2:M-1) ...
          +0.5*lambda1*((C(N+2,3:M)+C(N+1,3:M)).*U(N,2:M-1)...
          -(2*C(N+1,3:M)+C(N+2,3:M)+C(N,3:M)).*U(N,2:M-1)...
                      +(C(N,3:M)+C(N+1,3:M)).*U(N-1,2:M-1)) ...
          +0.5*lambda2*((C(N+1,4:M+1)+C(N+1,3:M)).*U(N,3:M)...
          -(C(N+1,4:M+1)+2*C(N+1,3:M)+C(N+1,2:M-1)).*U(N,2:M-1)...
            +(C(N+1,2:M-1)+C(N+1,3:M)).*U(N,1:M-2));

        U_old(N, 2:M-1)=U_old(N, 2:M-1) ;

        % j=1 update
         U_old(2:N-1, 1)=U(2:N-1, 1) ...
          +0.5*lambda1*((C(4:N+1,2)+C(3:N,2)).*U(3:N,1)...
          -(2*C(3:N,2)+C(4:N+1,2)+C(2:N-1,2)).*U(2:N-1,1)...
                      +(C(2:N-1,2)+C(3:N,2)).*U(1:N-2,1)) ...
          +0.5*lambda2*((C(3:N,3)+C(3:N,2)).*U(2:N-1,2)...
          -(C(3:N,3)+2*C(3:N,2)+C(3:N,1)).*U(2:N-1,1)...
            +(C(3:N,1)+C(3:N,2)).*U(2:N-1,1));

         U_old(2:N-1, 1)=U_old(2:N-1, 1) ;


        % j=M update
         U_old(2:N-1, M)=U(2:N-1, M) ...
          +0.5*lambda1*((C(4:N+1,M+1)+C(3:N,M+1)).*U(3:N,M)...
          -(2*C(3:N,M+1)+C(4:N+1,M+1)+C(2:N-1,M+1)).*U(2:N-1,M)...
                      +(C(2:N-1,M+1)+C(3:N,M+1)).*U(1:N-2,M)) ...
          +0.5*lambda2*((C(3:N,M+2)+C(3:N,M+1)).*U(2:N-1,M)...
          -(C(3:N,M+2)+2*C(3:N,M+1)+C(3:N,M)).*U(2:N-1,M)...
            +(C(3:N,M)+C(3:N,M+1)).*U(2:N-1,M-1));

        U_old(2:N-1, M)=U_old(2:N-1, M) ;


         % i=1, j=1 update
        U_old(1, 1)=U(1, 1) ...
          +0.5*lambda1*((C(3,2)+C(2,2)).*U(2,1)...
          -(2*C(2,2)+C(3,2)+C(1,2)).*U(1,1)...
                      +(C(1,2)+C(2,2)).*U(1,1)) ...
          +0.5*lambda2*((C(2,3)+C(2,2)).*U(1,2)...
          -(C(2,3)+2*C(2,2)+C(2,1)).*U(1,1)...
            +(C(2,2)+C(2,1)).*U(1,1));

        U_old(1, 1)=U_old(1, 1) ;

        % i=N, j=1 update
         U_old(N, 1)=U(N, 1) ...
          +0.5*lambda1*((C(N+2,2)+C(N+1,2)).*U(N,1)...
          -(2*C(N+1,2)+C(N+2,2)+C(N,2)).*U(N,1)...
                      +(C(N,2)+C(N+1,2)).*U(N-1,1)) ...
          +0.5*lambda2*((C(N+1,3)+C(N+1,2)).*U(N,2)...
          -(C(N+1,3)+2*C(N+1,2)+C(N+1,1)).*U(N,1)...
            +(C(N+1,2)+C(N+1,1)).*U(N,1));

        U_old(N, 1)=U_old(N, 1) ;  

        % j=M, i=1 update
         U_old(1, M)=U(1, M) ...
          +0.5*lambda1*((C(3,M+1)+C(2,M+1)).*U(2,M)...
          -(2*C(2,M+1)+C(3,M+1)+C(1,M+1)).*U(1,M)...
                      +(C(1,M+1)+C(2,M+1)).*U(1,M)) ...
          +0.5*lambda2*((C(2,M+2)+C(2,M+1)).*U(1,M)...
          -(C(2,M+2)+2*C(2,M+1)+C(2,M)).*U(1,M)...
            +(C(2,M+1)+C(2,M)).*U(1,M-1));
        U_old(1, M)=U_old(1, M) ;

        % j=M, i=N update
         U_old(N, M)=U(N, M) ...
          +0.5*lambda1*((C(N+2,M+1)+C(N+1,M+1)).*U(N,M)...
          -(2*C(N+1,M+1)+C(N+2,M+1)+C(N,M+1)).*U(N,M)...
                      +(C(N,M+1)+C(N+1,M+1)).*U(N-1,M)) ...
          +0.5*lambda2*((C(N+1,M+2)+C(N+1,M+1)).*U(N,M)...
          -(C(N+1,M+2)+2*C(N+1,M+1)+C(N+1,M)).*U(N,M)...
            +(C(N+1,M+1)+C(N+1,M)).*U(N,M-1));
        U_old(N, M)=U_old(N, M);

        U=U_old;
    end
    
    %update only in mask
    for i=1:N
       for j=1:M
           if (mask(i,j)==255)
               if U(i,j)>255
                   U(i,j)=255;
               end
               if U(i,j)<0
                    U(i,j)=0;
               end
               img(i,j) = U(i,j);
           end
       end
    end
end
subplot(1,3,1);imshow(uint8(oimg));title('original image');
subplot(1,3,2);imshow(uint8(img));title('inpainted image');
subplot(1,3,3);imshow(uint8(mask)); title('chosen mask');