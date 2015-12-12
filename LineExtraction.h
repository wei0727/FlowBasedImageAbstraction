#ifndef LINE_EXTRACTION
#define LINE_EXTRACTION

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <opencv2\opencv.hpp>

using namespace std;
using namespace cv;

void showMinMaxVal(Mat m, string s=""){
	double vmin, vmax;
	Point pmin, pmax;
	minMaxLoc(m, &vmin, &vmax, &pmin, &pmax);
	cout << s << vmin << "\t" << vmax << endl;
}

namespace npr{

	struct param{
		int ksize;
		double sigma_m;
		double sigma_c;
		double sigma_s;
		double lo;
		double tau;
		double phi;

		explicit param(int ks = 5, double sm = 3.0, double sc = 1.0, double ss = 1.6, double l = 0.99, double t = 0.85, double p = 10.0){
			ksize = ks;
			sigma_m = sm;
			sigma_c = sc;
			sigma_s = ss;
			lo = l;
			tau = t;
			phi = p;
		};
	};

	cv::Mat FGaussian(cv::Mat src, cv::Mat t, const int ksize, const double sigma_s, const double sigma_t);
	cv::Mat FDoG(cv::Mat src, cv::Mat t, param p);
	cv::Mat FDoG_myVer(cv::Mat src, cv::Mat t, param p);
	cv::Mat FDoG(cv::Mat src, cv::Mat t, param p, int iterations);

	///Implementation
	Mat FGaussian(cv::Mat src, cv::Mat t, const int ksize, const double sigma_s, const double sigma_t){
		const int rows = src.rows;
		const int cols = src.cols;
		const int L = 1.5*ksize;

		const double sigma_s2 = sigma_s*sigma_s;
		const double sigma_t2 = sigma_t*sigma_t;
		Mat out(rows, cols, CV_32F, Scalar(0));

		//Gaussian perpendicular to tangent
		Mat tmp = out.clone();
		for (int r = 0; r < rows; r++){
			for (int c = 0; c < cols; c++){
				Vec2f gi = t.at<Vec2f>(r, c);
				gi = Vec2f(gi[1], -gi[0]);
				double weight = 0;
				//move along gi
				for (int t = -ksize; t <= ksize; t++){
					int x = static_cast<int>(c + gi[0] * t);
					int y = static_cast<int>(r + gi[1] * t);
					if (x < 0 || x >= cols || y < 0 || y >= rows)
						continue;
					double w = exp(-t*t / sigma_t2);
					weight += w;
					tmp.at<float>(r, c) += static_cast<float>(w*src.at<float>(y, x));
				}
				tmp.at<float>(r, c) /= weight;
			}
		}

		//Gaussian along tangent
		for (int r = 0; r < rows; r++){
			for (int c = 0; c < cols; c++){
				double weight = 0;
				//two direction 
				for (int d = -1; d <= 1; d += 2){
					int l = 0;
					Vec2f ti = d * t.at<Vec2f>(r, c);
					Point cpt(c, r);
					while (++l < L){
						int px = static_cast<int>(round(cpt.x));
						int py = static_cast<int>(round(cpt.y));
						if (px < 0 || px >= cols || py<0 || py>=rows)
							break;

						double w = exp(-l*l / sigma_s2);
						weight += w;
						out.at<float>(r, c) += static_cast<double>(w*tmp.at<float>(py, px));

						//move to next point
						Vec2f v = d * t.at<Vec2f>(py, px);
						if (v == Vec2f(0, 0))
							break;

						//Check the direction by dot product, not sure if necessary
						//cpt += v.dot(ti) > 0 ? Point2d(v[0], v[1]) : -Point2d(v[0], v[1]);
						cpt = Point(round(px + v[0]), round(py + v[1]));
					}
				}
				//if (weight != 0)
					out.at<float>(r, c) /= weight;
				//else
				//	out.at<float>(r, c) = tmp.at<float>(r, c);
			}
		}

		return out;
	}

	cv::Mat FDoG(cv::Mat src, cv::Mat t, param p){
		const int rows = src.rows;
		const int cols = src.cols;
		Mat out(rows, cols, CV_32F, Scalar(1.0));

		cout << p.sigma_c << "\t" << p.sigma_s << endl;
		Mat g1 = npr::FGaussian(src, t, p.ksize, p.sigma_m, p.sigma_c);
		Mat g2 = npr::FGaussian(src, t, p.ksize, p.sigma_m, p.sigma_s);

		Mat d = g1 - p.lo*g2;
		cout << d.depth() << endl;
		showMinMaxVal(d, "npr::seperateVer: ");
		//Mat tmp_h;
		//normalize(d, tmp_h, 0.0, 1.0, CV_MINMAX);
		//imshow("seperate", tmp_h);

		threshold(d, d, 0, 0, THRESH_TRUNC);
		normalize(d, d, -CV_PI, 0.0, CV_MINMAX);

		for (int r = 0; r < rows; r++){
			float* p1 = g1.ptr<float>(r);
			float* p2 = g2.ptr<float>(r);
			for (int c = 0; c < cols; c++){
				double diff = p1[c] - p.lo*p2[c];
				if (diff < 0 && (1 + tanh(d.at<float>(r,c))) < 0.7)
					out.at<float>(r, c) = 0;
			}
		}	

		return out;
	}

	cv::Mat FDoG_myVer(cv::Mat src, cv::Mat t, param p){
		const int rows = src.rows;
		const int cols = src.cols;

		//const double alpha = p.sigma_m * 3.0;
		//const double beta = p.sigma_s * 3.0;
		const double alpha = 5*1.5;
		const double beta = 5;

		const double sigma_m2 = 2 * p.sigma_m*p.sigma_m;
		const double sigma_c2 = 2 * p.sigma_c*p.sigma_c;
		const double sigma_s2 = 2 * p.sigma_s*p.sigma_s;
		const double g_sigma_m = 1.0 / (sqrt(2.0*CV_PI)*p.sigma_m);
		const double g_sigma_c = 1.0 / (sqrt(2.0*CV_PI)*p.sigma_c);
		const double g_sigma_s = 1.0 / (sqrt(2.0*CV_PI)*p.sigma_s);

		//Gaussian along gradient 
		Mat tmp(rows, cols, CV_32F, Scalar(0));
		for (int r = 0; r < rows; r++){
			for (int c = 0; c < cols; c++){
				Vec2f ti = t.at<Vec2f>(r, c);
				Vec2f gi(ti[1], -ti[0]);
				double g1 = 0, g2 = 0;
				double weight1 = 0, weight2 = 0;
				//d for direction
				for (int d = -1; d <= 1; d += 2){
					Point2d cpt(c, r);
					for (int i = 0; i < beta; i++){
						int px = round(cpt.x);
						int py = round(cpt.y);
						if (px < 0 || py < 0 || px >= cols || py >= rows)
							break;

						double w = g_sigma_c*exp(-(i*i) / sigma_c2);
						double w2 = g_sigma_s*exp(-(i*i) / sigma_s2);
						//w = w - p.lo*w2;
						if (d == 1 && px == c && py == r);
						else{
							//tmp.at<float>(r, c) += src.at<float>(py, px) * w;
							g1 += src.at<float>(py, px) * w;
							g2 += src.at<float>(py, px) * w2;
							weight1 += w;
							weight2 += w2;
						}
						if (gi == Vec2f(0, 0))
							break;
					
						Point2d npt = cpt + d*Point2d(gi[0], gi[1]);
						//int npx = round(npt.x);
						//int npy = round(npt.y);
						//while (npx == px && npy == py){
						//	npt += d*Point2d(gi[0], gi[1]);
						//	npx = round(npt.x);
						//	npy = round(npt.y);
						//}

						cpt = npt;
					}
				}
				tmp.at<float>(r, c) = g1 / weight1 - p.lo*g2 / weight2;
			}
		}

		//Gaussian along tangent
		Mat h(rows, cols, CV_32F, Scalar(0));
		for (int r = 0; r < rows; r++){
			for (int c = 0; c < cols; c++){
				double weight = 0;
				for (int d = -1; d <= 1; d += 2){
					Point2d cpt(c, r);
					for (int i = 0; i < alpha; i++){
						int px = round(cpt.x);
						int py = round(cpt.y);
						if (px < 0 || py < 0 || px >= cols || py >= rows)
							break;

						//if (d != 1 || cpt == Point2d(c, r)){
							double w = g_sigma_m * exp(-i*i / sigma_m2);
							if (d == 1 && px == c && py == r);
							else{
								h.at<float>(r, c) += tmp.at<float>(py, px) * w;
								weight += w;
							}
						//}

						Vec2f ti = t.at<Vec2f>(py, px);
						if (ti == Vec2f(0, 0))
							break;

						Point2d npt = cpt + d*Point2d(ti[0], ti[1]);
						//int npx = round(npt.x);
						//int npy = round(npt.y);
						//while (npx == px && npy == py){
						//	npt += d*Point2d(ti[0], ti[1]);
						//	npx = round(npt.x);
						//	npy = round(npt.y);
						//}

						cpt = npt;
					}
				}
				h.at<float>(r, c) /= weight;
			}
		}

		//showMinMaxVal(h, "npr::myver: ");
		//Mat tmp_h;
		//normalize(h, tmp_h, 0.0, 1.0, CV_MINMAX);
		//imshow("myver", tmp_h);
		//Thresholding
		threshold(h, h, 0, 0, THRESH_TRUNC);
		normalize(h, h, -CV_PI, 0.0, CV_MINMAX);

		Mat edge(rows, cols, CV_32F, Scalar(1.0));
		for (int r = 0; r < rows; r++){
			float* ph = h.ptr<float>(r);
			for (int c = 0; c < cols; c++){
				if (ph[c] < 0 && (1 + tanh(ph[c])) < p.tau)
					edge.at<float>(r, c) = 0;
			}
		}
		return edge;
	}

	cv::Mat FDoG(cv::Mat src, cv::Mat t, param p, int iterations){
		const int rows = src.rows;
		const int cols = src.cols;

		npr::param mp;
		Mat tmp = src.clone();
		Mat edge;
		while (iterations--){
			edge = npr::FDoG_myVer(tmp, t, p);
			//edge = npr::FDoG(tmp, t, p);
			for (int r = 0; r < rows; r++){
				for (int c = 0; c < cols; c++){
					if (edge.at<float>(r, c) == 0)
						tmp.at<float>(r, c) = 0;
				}
			}
			imshow("iterations", edge);
			waitKey();
		}

		return edge;
	}
}

#endif