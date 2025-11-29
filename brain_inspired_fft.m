clc;
clear;
close all;


function y = sample(input, freq, outputsize, outfreq)
    list = zeros(1,outputsize);
    step_rate = freq/outfreq;
    for N = 1:outputsize
        list(N) = input(floor(step_rate*N));
    end
    y= list;
end

%part 1
function y = dft_recursive(input)
    s = int32(size(input,2));
    if s == 1
        y = [input(1)];
    else
        o_list = zeros(1,s/2);
        e_list = zeros(1,s/2);
        for i = 1:s
            if mod(i,2) ==1
                e_list((i+1)/2) = input(i);
            else
                o_list(i/2) = input(i);
            end
        end
        dft_o = dft_recursive(o_list);
        dft_e = dft_recursive(e_list);
        dft = zeros(1,s);
         for b = 1:s/2
            dft(b) = dft_e(b) + exp(-1i*2*pi*(double(b)-1)/double(s)) * dft_o(b);
            dft(b+s/2) = dft_e(b) - exp(-1i*2*pi*(double(b)-1)/double(s)) * dft_o(b);
         end
        %end
    y = dft;
    end
    
end

function y = dft_recursive_brain(input, theta)
    s = int32(size(input,2));
    if s == 1
        y = [input(1)];
    else
        o_list = zeros(1,s/2);
        e_list = zeros(1,s/2);
        for i = 1:s
            if mod(i,2) ==1
                e_list((i+1)/2) = input(i);
            else
                o_list(i/2) = input(i);
            end
        end
        dft_o = dft_recursive_brain(o_list, theta);
        dft_e = dft_recursive_brain(e_list, theta);
        dft = zeros(1,s);
         for b = 1:s/2
            theta_local = mod((b-1), length(theta)) + 1;
            dft(b) = dft_e(b) + exp(-1i*2*pi*(double(b)-1)/double(s) + theta(theta_local)) * dft_o(b);
            dft(b+s/2) = dft_e(b) - exp(-1i*2*pi*(double(b)-1)/double(s) + theta(theta_local)) * dft_o(b);
         end
        %end
    y = dft;
    end
    
end

function y = dft_iterative(input)
    n = int32(round(log2(size(input,2))));
    inp_ = bitrevorder(input);
    for i = 1:n
        subn = int32(2^i);
        list = zeros(1,size(input,2));
        %w = e^(-2*pi*1i*1/subn);
        for j = 1:subn:size(input,2)
            w=1.00;
            for k = 1:subn/2
                list((j-1) + k) = inp_((j-1) + k) + w* inp_((j-1) + k+subn/2);
                list((j-1) + k + subn/2) = inp_((j-1) + k) - w* inp_((j-1) + k+subn/2);
                w = w* exp(-2*pi*1i*1/double(subn));
            end
        end
        inp_ = list;

    end
    y =  inp_;
end

function y = dft_iterative_brain(input, theta)
    n = int32(round(log2(size(input,2))));
    inp_ = bitrevorder(input);
    for i = 1:n
        subn = int32(2^i);
        list = zeros(1,size(input,2));
        %w = e^(-2*pi*1i*1/subn);
        for j = 1:subn:size(input,2)
            w=1.00;
            for k = 1:subn/2
                theta_local = mod((k-1)*(n/subn), length(theta)) + 1;
                list((j-1) + k) = inp_((j-1) + k) + w* inp_((j-1) + k+subn/2);
                list((j-1) + k + subn/2) = inp_((j-1) + k) - w* inp_((j-1) + k+subn/2);
                w = w* exp(-2*pi*1i*1/double(subn) + theta(theta_local));
            end
        end
        inp_ = list;

    end
    y =  inp_;
end

function z = fourier(input)
    frequency = linspace(0,1,size(input,2));
    s = size(input,2);
    sum =zeros(1,size(input,2));
    for d = 1:s
        sum = sum + input(d) * exp(-2*pi*1i*d*frequency);
    end
    z = sum;
end

function f = fft_cost(samp, x_true, theta)
        x_model1 = dft_iterative_brain(samp, theta);
        diff1 = x_model1 - x_true;
        x_model2 = dft_recursive_brain(samp, theta);
        diff2 = x_model2 - x_true;
        f = (sum(abs(diff1.^2))/(2*length(samp)) + sum(abs(diff2.^2))/(2*length(samp)))/2;
end

function dj_dw = fft_grad(samp, x_true, theta)
       eps = 1e-6;
    dj_dw = zeros(size(theta));
    for k = 1:length(theta)
        thp = theta; 
        thm = theta;
        thp(k) = thp(k)+eps;
        thm(k) = thm(k)-eps;
        dj_dw(k) = (fft_cost(samp,x_true,thp) - fft_cost(samp,x_true,thm)) / (2*eps);
    end
end

function [theta, J_hist] = gradient_descent_fft(samp, x_true, theta_init, alpha, num_iters)
       theta = theta_init;
       J_hist = zeros(1, num_iters);

       for iter = 1:num_iters
           grad = fft_grad(samp, x_true, theta);
           theta = theta - alpha*grad;
           J_hist(iter) = fft_cost(samp, x_true, theta);
       end

end

%part 2
time = 0:0.0001:15;
timefunc= 10*cos(2*pi*1e3*time) + 6*cos(2*pi*2e3*time)+2*cos(2*pi*4e3*time);
samp = sample(timefunc,1/0.001,64,1/0.003);
const = int32(size(samp,2));

x_true = fft(samp);
theta_initial = zeros(1, length(samp));
alpha = 0.01;
num_iters = 200;
[theta_final, J_hist] = gradient_descent_fft(samp, x_true, theta_initial, alpha, num_iters);
x_rec = dft_recursive(samp);
x_rec_brain = dft_recursive_brain(samp, theta_final);
x_it = dft_iterative(samp);
x_it_brain = dft_iterative_brain(samp, theta_final);
x_naive = fourier(samp);

figure();
figure(1);
plot(samp);
title("Signal in Time");

x_dot = fft(samp);
figure(2);
plot(fftshift(abs(x_dot)));
title("FFT via MATLAB function");

figure(3);
plot(fftshift(abs(x_rec)));
title("FFT via recursive function");

figure(4);
plot(fftshift(abs(x_rec_brain)));
title("FFT via brain inspired recursive function");

figure(5);
plot(fftshift(abs(x_it)));
title("FFT via iterative function");

figure(6);
plot(fftshift(abs(x_it_brain)));
title("FFT via brain inspired iterative function");

x_naive = fourier(samp);
figure(7);
plot(fftshift(abs(x_naive)));
title("FFT via naive function");

SQ = abs(x_dot* x_dot.');
disp("Error in recursive");
immse(x_dot,x_rec)/SQ
disp("Error in brain inspired recursive");
immse(x_dot,x_rec_brain)/SQ
disp("Error in iterative");
immse(x_dot,x_it)/SQ
disp("Error in brain inspired iterative");
immse(x_dot,x_it_brain)/SQ

%part 3
L=10;
numbers = 1:L;
numbers = 2.^numbers;
t_naive = zeros(1,L);
t_rec = zeros(1,L);
t_it = zeros(1,L);
t_matl = zeros(1,L);
for i = 1:L
    samp = sample(timefunc,1/0.001,numbers(i),1/0.003);
    tic;
    z = fft(samp);
    t_matl(i) = toc;
    tic;
    z = fourier(samp);
    t_naive(i) = toc;
    tic;
    z = dft_recursive(samp);
    t_rec(i) = toc;
    tic;
    z = dft_recursive_brain(samp, theta_final);
    t_rec_brain(i) = toc;
    tic;
    z = dft_iterative(samp);
    t_it(i) = toc;
    z = dft_iterative_brain(samp, theta_final);
    t_it_brain(i) = toc;
end
figure(8);
hold on;
plot(1:L,t_rec);
plot(1:L,t_rec_brain);
plot(1:L,t_matl);
plot(1:L,t_naive);
plot(1:L,t_it);
plot(1:L,t_it_brain);
legend("recursive","brain inspired recursive","matlab","naive","iterative","brain inspired iterative");
title("time comparison of fft functions");
hold off;