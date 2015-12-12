#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <opencv2\opencv.hpp>

#include "LIC.h"
#include "LineExtraction.h"

using namespace std;
using namespace cv;

//#define DBG

//Tools
void LICImage(vector<vector<Vec2d>> src, int n_xres, int n_yres, vector<vector<float>> m){
	//float* pVectr = (float*)malloc(sizeof(float)* n_xres * n_yres * 2);
	float* pVectr = new float[n_xres*n_yres*2];
	for (int r = 0; r < n_yres; r++){
		for (int c = 0; c < n_xres; c++){
			int	 index = ((r) * n_xres + c) << 1;
			//pVectr[index] = src[r][c][0]*m[r][c];
			//pVectr[index + 1] = src[r][c][1]*m[r][c];
			pVectr[index] = src[r][c][0];
			pVectr[index + 1] = src[r][c][1];
		}
	}
	unsigned char* licArr = new unsigned char[n_xres*n_yres];
	LIC(n_xres, n_yres, pVectr, licArr);
	Mat licImg(n_yres, n_xres, CV_8U, licArr);
	//resize(licImg, licImg, Size(800, 800), 0, 0, CV_INTER_CUBIC);
	imshow("licImg", licImg);
	waitKey();
	destroyWindow("licImg");

	delete[] pVectr;
	delete[] licArr;
}

//Edge tangent filter
//Input a gray image (L channel of Lab Image)
//ETF for t times
Mat ETF(Mat& src, int t){
	//Initialize
	Mat g0x, g0y;
	vector<vector<float>> g0m(src.rows, vector<float>(src.cols));
	Sobel(src, g0x, CV_32F, 1, 0, 3, 1, 0, BORDER_DEFAULT);
	Sobel(src, g0y, CV_32F, 0, 1, 3, 1, 0, BORDER_DEFAULT);

#ifdef DBG
	Mat g0x_abs, g0y_abs;
	convertScaleAbs(g0x, g0x_abs);
	convertScaleAbs(g0y, g0y_abs);
	imshow("g0x", g0x_abs);
	imshow("g0y", g0y_abs);
	waitKey();
	destroyWindow("g0x");
	destroyWindow("g0y");
#endif 

	vector<vector<Vec2d>> t0(src.rows, vector<Vec2d>(src.cols));
	Mat g0mImg(src.size(), CV_32F);
	for (int r = 0; r < src.rows; r++){
		float* pgx = g0x.ptr<float>(r);
		float* pgy = g0y.ptr<float>(r);
		float* pgm = g0mImg.ptr<float>(r);
		for (int c = 0; c < src.cols; c++){
			Vec2d g(pgx[c], pgy[c]);
			float gm = norm(g);
			g0m[r][c] = gm;
			pgm[c] = g0m[r][c];
			if (gm > 0)
				g /= gm;
			//rotate gradient vector 90 degree
			t0[r][c][0] = -g[1];
			t0[r][c][1] = g[0];
		}
	}
	normalize(g0mImg, g0mImg, 0.0, 1.0, NORM_MINMAX);
	for (int r = 0; r < src.rows; r++){
		float* p = g0mImg.ptr<float>(r);
		for (int c = 0; c < src.cols; c++)
			g0m[r][c] = p[c];
	}

#ifdef DBG
	imshow("g0m", g0mImg);
	waitKey();
	destroyWindow("g0m");
	//LICImage(t0, src.cols, src.rows, g0m);
#endif

	//Filter for t times
	//windows size
	const int ws = 2;	
	vector<vector<Vec2d>> t1(src.rows, vector<Vec2d>(src.cols));
	for (int i = 0; i < t; i++){
		for (int r = 0; r < src.rows; r++){
			for (int c = 0; c < src.cols; c++){
				int r0 = max(0, r - ws);
				int r1 = min(src.rows - 1, r + ws);
				int c0 = max(0, c - ws);
				int c1 = min(src.cols - 1, c + ws);
				Vec2f tnew(0, 0);
				int count = 0;
				for (int rw = r0; rw <= r1; rw++){
					for (int cw = c0; cw <= c1; cw++){
						if (norm(Vec2f(r, c) - Vec2f(rw, cw)) <= ws){
							float wm = (g0m[rw][cw] - g0m[r][c] + 1) / 2.0;
							float wd = t0[r][c].dot(t0[rw][cw]);
							float fi = wd > 0 ? 1 : -1;
							wd = abs(wd);
							tnew += wm*wd*fi*t0[rw][cw];
							count++;
						}
					}
				}
				//t1[r][c] = tnew/(float)count;
				t1[r][c] = normalize(tnew);
			}
		}
		t0 = t1;

#ifdef DBG
		cout << i << endl;
 		LICImage(t0, src.cols, src.rows, g0m);
#endif
	} 

	int count_zero = 0;
	Mat t_final(src.size(), CV_32FC2);
	for (int r = 0; r < src.rows; r++){
		Vec2f* p = t_final.ptr<Vec2f>(r);
		for (int c = 0; c < src.cols; c++){
			p[c] = t0[r][c];
			if (p[c] == Vec2f(0, 0))
				count_zero++;
		}
	}
	cout << "zero count: " << count_zero << endl;

	return t_final;
}

//Flow based DoG filter
//input L image and its edge tangent field
Mat FDoG(Mat src, Mat t){
	const double sigma_m = 3.0;
	const double sigma_c = 1.0;
	const double sigma_s = 1.6 * sigma_c;
	const double lo = 0.99;

	//for compute
	//const int alpha = 3 * sigma_m;
	//const int beta = 3 * sigma_s;
	const int alpha = 5*1.5;
	const int beta = 5;
	const double g_sigma_m = 1.0 / (sqrt(2.0*CV_PI)*sigma_m);
	const double g_sigma_c = 1.0 / (sqrt(2.0*CV_PI)*sigma_c);
	const double g_sigma_s = 1.0 / (sqrt(2.0*CV_PI)*sigma_s);

	//precompute
	vector<float> gs(2 * alpha + 1);
	for (int i = 0; i <= 2 * alpha; i++){
		float s = i <= alpha ? i: alpha - i;
		gs[i] = g_sigma_m*exp(-(s*s) / (2.0 * sigma_m*sigma_m));
#ifdef DBG
		cout << "s: " << s << "\tgs: " << gs[i] << endl;
#endif
	}
	vector<float> ft(2 * beta + 1);
	for (int i = 0; i <= 2 * beta; i++){
		float t = i <= beta ? i : beta - i;
		ft[i] = g_sigma_c*exp(-(t*t) / (2.0 * sigma_c*sigma_c)) - lo*g_sigma_s*exp(-(t*t) / (2.0 * sigma_s*sigma_s));
#ifdef DBG
		cout << "t: " << t << "\tft: " << ft[i] << endl;
#endif
	}

#ifdef DBG
	cout << "gs_sum: " << sum(gs) << "\nft_sum: " << sum(ft) << endl;
#endif

	Mat h(src.size(), CV_32F);
	for (int r = 0; r < t.rows; r++){
		for (int c = 0; c < t.cols; c++){
			Vec2f t0 = t.at<Vec2f>(r, c);
			//If tangent = (0, 0)
			if (t0 == Vec2f(0, 0)){
				h.at<float>(r, c) = src.at<float>(r, c)*(g_sigma_c - lo*g_sigma_s)*g_sigma_m;
			}
			else{
				vector<vector<float>> curve(2*alpha+1, vector<float>(2*beta+1));
				Point cur(c, r);
				float he = 0;
				//i, travel through the curve center at (c, r)
				for (int i = 0; i <= 2 * alpha; i++){
					//check if current postion out of image
					if (cur.x < 0 || cur.x >= src.cols || cur.y < 0 || cur.y >= src.rows){
						//if in positive step, transfer to negative step
						if (i <= alpha){
							i = alpha;
							Vec2f ti = t.at<Vec2f>(Point(c, r));
							cur = Point(round(c - ti[0]), round(r - ti[1]));
							continue;
						}
						//if in negative step, end of the travel
						else
							break;
					}
					float hg = 0;
					Vec2f ti = t.at<Vec2f>(cur);
					Vec2f gi(ti[1], -ti[0]);
					if (ti == Vec2f(0, 0))
						break;
					Point tmp = cur;
					//travel through the perpendicular line 
					for (int j = 0; j <= 2 * beta; j++){
						//check if current point out of image
						if (tmp.x < 0 || tmp.x >= src.cols || tmp.y < 0 || tmp.y >= src.rows){
							if (j <= beta){
								j = beta;
								tmp = Point(round(cur.x - gi[0]), round(cur.y - gi[1]));
								continue;
							}
							else
								break;
						}
						curve[i][j] = src.at<float>(tmp) * ft[j];
						hg += curve[i][j];
						if (j < beta)
							tmp = Point(round(tmp.x + gi[0]), round(tmp.y + gi[1]));
						else{
							if (j == beta)
								tmp = cur;
							tmp = Point(round(tmp.x - gi[0]), round(tmp.y - gi[1]));
						}
					}
					he += hg * gs[i];
					if (i < alpha)
						cur = Point(round(cur.x + ti[0]), round(cur.y + ti[1]));
					else{
						if (i == alpha){
							cur = Point(c, r);
							ti = t.at<Vec2f>(cur);
						}
						cur = Point(round(cur.x - ti[0]), round(cur.y - ti[1]));
					}
				}
				h.at<float>(r, c) = he;
			}
		}
	}

	showMinMaxVal(h, "originalVer: ");
	//Mat tmp_h;
	//normalize(h, tmp_h, 0.0, 1.0, CV_MINMAX);
	//imshow("orig", tmp_h);

	const double tau = 0.85;
	Mat h_tiled(src.size(), CV_32F);

	Mat h_tau(src.size(), CV_32F);
	Mat h_th1(src.size(), CV_32F);
	Mat h_th2(src.size(), CV_32F);
	//NOT SURE...
	//Different with paper
	//only h<0, 1 threshold, due to the scale of h are very small-->1+tanh very large
	//Normalize h values
	threshold(h, h, 0, 0, THRESH_TRUNC);
	normalize(h, h, -CV_PI, 0.0, CV_MINMAX);
	//
	for (int r = 0; r < src.rows; r++){
		float* p = h.ptr<float>(r);
		float* pt = h_tiled.ptr<float>(r);
		for (int c = 0; c < src.cols; c++){
			if (p[c] < 0 && (1 + tanh(p[c])) < tau)
			//if (p[c] < 0)
				pt[c] = 0;
			else
				pt[c] = 1;

			h_th1.at<float>(r, c) = p[c] < 0 ? 0 : 1;
			h_tau.at<float>(r, c) = 1 + tanh(p[c]);
			h_th2.at<float>(r, c) = h_tau.at<float>(r, c) < tau ? 0 : 1;
		}
	}

#ifdef DBG
	double vmax, vmin;
	Point pmax, pmin;
	minMaxLoc(h, &vmin, &vmax, &pmin, &pmax);
	cout << "h: " << vmin << "\t" << vmax << endl;

	minMaxLoc(h_tau, &vmin, &vmax, &pmin, &pmax);
	cout << "1+tanh: " << vmin << "\t" << vmax << endl;
	imshow("h_th1", h_th1);
	imshow("h_th2", h_th2);
	imshow("h_tiled", h_tiled);
	waitKey();
	destroyAllWindows();
#endif

	return h_tiled;
}

//Flow based Bilateral Filter
Mat FBL(Mat src, Mat t, int iterations){
	const double sigma_e = 1.0;
	const double re = 10 / 255.0;
	const double sigma_g = 0.5;
	const double rg = 10 / 255.0;

	//for compute
	const int alpha = 3 * sigma_e;
	const int beta = alpha;
	const double g_sigma_e = 1.0 / (sqrt(2.0*CV_PI)*sigma_e);
	const double g_re = 1.0 / (sqrt(2.0*CV_PI)*re);
	const double g_sigma_g = 1.0 / (sqrt(2.0*CV_PI)*sigma_g);
	const double g_rg = 1.0 / (sqrt(2.0*CV_PI)*rg);

	//precompute
	vector<float> gs(alpha * 2 + 1);
	vector<float> gt(alpha * 2 + 1);
	for (int i = 0; i < alpha * 2 + 1; i++){
		float s = i <= alpha ? i : alpha - i;
		gs[i] = g_sigma_e*exp(-(s*s) / (2.0 * sigma_e*sigma_e));
		gt[i] = g_sigma_g*exp(-(s*s) / (2.0 * sigma_g*sigma_g));
	}

	Mat h(src.size(), CV_32F);
	for (int it = 0; it < iterations; it++){
		for (int r = 0; r < t.rows; r++){
			for (int c = 0; c < t.cols; c++){
				Vec2f t0 = t.at<Vec2f>(r, c);
				//If tangent = (0, 0)
				if (t0 == Vec2f(0, 0)){
					//h.at<float>(r, c) = src.at<float>(r, c)*(g_sigma_c - lo*g_sigma_s)*g_sigma_m;
					h.at<float>(r, c) = src.at<float>(r, c);
				}
				else{
					vector<vector<float>> curve(2 * alpha + 1, vector<float>(2 * beta + 1));
					vector<float> cg_arr(2 * alpha + 1);
					Point cur(c, r);
					float ve = 0;
					float ce = 0;
					//i, travel through the curve center at (c, r)
					for (int i = 0; i <= 2 * alpha; i++){
						//check if current postion out of image
						if (cur.x < 0 || cur.x >= src.cols || cur.y < 0 || cur.y >= src.rows){
							//if in positive step, transfer to negative step
							if (i <= alpha){
								i = alpha;
								Vec2f ti = t.at<Vec2f>(Point(c, r));
								cur = Point(round(c - ti[0]), round(r - ti[1]));
								continue;
							}
							//if in negative step, end of the travel
							else
								break;
						}
						float vg = 0;
						float cg = 0;
						Vec2f ti = t.at<Vec2f>(cur);
						Vec2f gi(ti[1], -ti[0]);
						Point tmp = cur;
						//travel through the perpendicular line 
						for (int j = 0; j <= 2 * beta; j++){
							//check if current point out of image
							if (tmp.x < 0 || tmp.x >= src.cols || tmp.y < 0 || tmp.y >= src.rows){
								if (j <= beta){
									j = beta;
									tmp = Point(round(cur.x - gi[0]), round(cur.y - gi[1]));
									continue;
								}
								else
									break;
							}
							curve[i][j] = gt[j] * (g_rg*exp(-(abs(src.at<float>(tmp)-src.at<float>(cur)) / 2 * rg*rg)));
							vg += curve[i][j];
							cg += src.at<float>(tmp) * curve[i][j];
							if (j < beta)
								tmp = Point(round(tmp.x + gi[0]), round(tmp.y + gi[1]));
							else{
								if (j == beta)
									tmp = cur;
								tmp = Point(round(tmp.x - gi[0]), round(tmp.y - gi[1]));
							}
						}

						cg_arr[i] = cg / vg;
						float tmp_ve = g_sigma_e * (g_re*exp(-(abs(cg_arr[i] - cg_arr[0]) / 2 * re*re)));
						ve += tmp_ve;
						ce += cg_arr[i] * tmp_ve;


						if (i < alpha)
							cur = Point(round(cur.x + ti[0]), round(cur.y + ti[1]));
						else{
							if (i == alpha){
								cur = Point(c, r);
								ti = t.at<Vec2f>(cur);
							}
							cur = Point(round(cur.x - ti[0]), round(cur.y - ti[1]));
						}
					}
					h.at<float>(r, c) = ce / ve;
				}
			}
		}
		src = h.clone();
	}

#ifdef DBG
	imshow("src", src);
	imshow("filtered result", h);
	waitKey();
	destroyAllWindows();
#endif

	return h;
}

Mat color_quantization(Mat src, int k){
	float step = 1.0 / k;
	vector<float> values(k);
	for (int i = 0; i < k; i++){
		values[i] = step*(i + 0.5);
	}
	Mat m(src.size(), CV_32F);
	for (int r = 0; r < src.rows; r++){
		float* ps = src.ptr<float>(r);
		float* pm = m.ptr<float>(r);
		for (int c = 0; c < src.cols; c++){
			int n = floor(ps[c] / step);
			if (n == k)
				n -= 1;
			pm[c] = values[n];
		}
	}
	return m;
}

int main(){
	string path = "cat.jpg";
	Mat img = imread("Img/"+path);
	Mat img_lab;
	vector<Mat> lab_channel;
	Mat L_32f;
	
	cvtColor(img, img_lab, COLOR_BGR2Lab);
	split(img_lab, lab_channel);
	lab_channel[0].convertTo(L_32f, CV_32F, 1.0 / 255.0);
	
	Mat img_blur;
	GaussianBlur(lab_channel[0], img_blur, Size(5, 5), 0, 0);
	//medianBlur(lab_channel[0], img_blur, 5);
	Mat t = ETF(img_blur, 3);

	GaussianBlur(L_32f, img_blur, Size(3, 3), 0, 0);
	//medianBlur(L_32f, img_blur, 5);
	//Mat src_lines = FDoG(img_blur, t);
	Mat src_lines = FDoG(L_32f, t);

	Mat L_fbl = FBL(L_32f, t, 3);
	Mat L_qt = color_quantization(L_fbl, 6);

	L_qt.convertTo(L_qt, CV_8U, 255);
	lab_channel[0] = L_qt;
	Mat src_fbl;
	merge(lab_channel, src_fbl);
	cvtColor(src_fbl, src_fbl, COLOR_Lab2BGR);

	Mat result = src_fbl.clone();
	for (int r = 0; r < img.rows; r++){
		float* pl = src_lines.ptr<float>(r);
		for (int c = 0; c < img.cols; c++){
			if (pl[c] == 0)
				result.at<Vec3b>(r, c) = Vec3b(0, 0, 0);
		}
	}

	npr::param mp;
	Mat edge = npr::FDoG(L_32f, t, mp);
	Mat edge2 = npr::FDoG_myVer(L_32f, t, mp);
	Mat edge3 = npr::FDoG(L_32f, t, mp, 3);

	imshow("src", img);
	imshow("lines", src_lines);
	//imshow("quantization", src_fbl);
	//imshow("result", result);
	imshow("edge", edge);
	imshow("edge2", edge2);
	waitKey();
	//imwrite("Outputs/"+path, result);

	//system("pause");
	return 0;
}