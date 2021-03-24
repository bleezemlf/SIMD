#include <iostream>
#include <iomanip>
#include <thread>
#include <vector>
#include <chrono>
#include <immintrin.h>
using namespace std;

class MatMult
{
protected:
	float ** _m1;
	float ** _m2;
	int  _size;
	float ** _r;
public:
	MatMult(int size , float value)
	{
		_size = size;
		_m1 = new float* [_size];
		for (int i = 0; i < _size; i++)
		{
			_m1[i] = new float[_size];
		}
		for (int i = 0; i < _size; i++)
		{
			for (int j = 0; j < _size; j++)
			{
				_m1[i][j] = value;
			}
		}
		_m2 = new float* [_size];
		for (int i = 0; i < _size; i++)
		{
			_m2[i] = new float[_size];
		}
		for (int i = 0; i < _size; i++)
		{
			for (int j = 0; j < _size; j++)
			{
				_m2[i][j] = value;
			}
		}
		_r = new float* [_size];
		for (int i = 0; i < _size; i++)
		{
			_r[i] = new float[_size];
		}
		for (int i = 0; i < _size; i++)
		{
			for (int j = 0; j < _size; j++)
				_r[i][j] = 0;
		}
	}

	virtual ~MatMult()
	{
		for (int i = 0; i < _size; i++) {
			delete[] _m1[i];
		}
		delete[] _m1;
		for (int i = 0; i < _size; i++) {
			delete[] _m2[i];
		}
		delete[] _m2;
		for (int i = 0; i < _size; i++) {
			delete[] _r[i];
		}
		delete[] _r;
	}

	virtual void matMult() = 0;
	
	void matShowR()
	{
		for (int i = 0; i < _size; i++)
		{
			for (int j = 0; j < _size; j++)
			{
				cout << _r[i][j];
			}
			cout << endl;
		}
	}

	void matTransM2()
	{
		float t;
		for (int i = 0; i < _size; i++)
		{
			for (int j = 0; j < i; j++)
			{
				t = _m2[i][j];
				_m2[i][j] = _m2[j][i];
				_m2[j][i] = t;
			}
		}
	}

	void resetR()
	{
		for (int i = 0; i < _size; i++)
		{
			for (int j = 0; j < _size; j++)
				_r[i][j] = 0;
		}
	}
};
	
class MatMultCommon :public MatMult
{
public:
	using MatMult::MatMult;
	void matMult()
	{
		auto startT = std::chrono::steady_clock::now();
		for (int i = 0; i < _size; i++)
		{
			for (int j = 0; j < _size; j++)
			{
				for (int n = 0; n < _size; n++)
				{
					_r[i][j] += _m1[i][n] * _m2[n][j];
				}
			}
		}
		auto endT = std::chrono::steady_clock::now();
		std::chrono::duration<double, std::micro> elapsed = endT - startT;
		cout << right << setw(5)
			<< "1" << '\t' << _size
			<< setw(11) << fixed << setprecision(3)
			<< (double)elapsed.count() / 1000 << "ms" << endl;
	}
};

class MatMultAlg :public MatMult
{
public:
	using MatMult::MatMult;
	void matMult()
	{
		matTransM2();
		auto startT = std::chrono::steady_clock::now();
		for (int i = 0; i < _size; ++i)
		{
			for (int j = 0; j < _size; ++j)
			{
				float temp = 0;
				for (int k = 0; k < _size; ++k)
					temp += _m1[i][k] * _m2[j][k];
				_r[i][j] = temp;
			}
		}
		auto endT = std::chrono::steady_clock::now();
		std::chrono::duration<double, std::micro> elapsed = endT - startT;
		cout << right << setw(5)
			<< '1' << '\t' << _size
			<< setw(11) << fixed << setprecision(3)
			<< (double)elapsed.count() / 1000 << "ms" << endl;
	}
};

class MatMultThreads: public MatMult
{
public:
	using MatMult::MatMult;

	void matMult(){}

	void matMult(int start, int cnt)
	{
		for (int i = start; i < start + cnt; ++i)
		{
			for (int j = 0; j < _size; j++)
			{
				for (int n = 0; n < _size; n++)
				{
					_r[i][j] += _m1[i][n] * _m2[n][j];
				}
			}
		}
	}

	void matMultTh()
	{
		matTransM2();
		auto f = [&](int start, int cnt) {matMult(start, cnt); };
		vector<thread> th;
		for (int i = 1; i <= 8; i++)
		{
			int step = _size / i;
			resetR();
			vector<thread> th;
			auto startT = std::chrono::steady_clock::now();
			for (int j = 0; j < i; j++)
			{
				if (j == i - 1)
					th.push_back(thread(f, j * step, _size - j * step));
				else
					th.push_back(thread(f, j * step, step));
			}
			for (auto& i : th)
				i.join();
			auto endT = std::chrono::steady_clock::now();
			std::chrono::duration<double, std::micro> elapsed = endT - startT;
			cout << right << setw(5)
				<< i << '\t' << _size
				<< setw(11) << fixed << setprecision(3)
				<< (double)elapsed.count() / 1000 << "ms" << endl;
		}
	}
};

class MatMultSIMD :public MatMult
{
protected:
	__m256** _mm1;
	__m256** _mm2;
public:
	MatMultSIMD(int size,float value) :MatMult(size, value)
	{
		size = _size / 8;

		_mm1 = new __m256*[_size];
		for (int i = 0; i < _size; i++)
		{
			_mm1[i] = new __m256[size];
		}
		size = _size / 8;
		for (int i = 0; i < _size; i++)
		{
			for (int j = 0; j < size; j++)
			{
				_mm1[i][j] = _mm256_loadu_ps(&_m1[i][j * 8]);
			}
		}

		_mm2 = new __m256* [_size];
		for (int i = 0; i < _size; i++)
		{
			_mm2[i] = new __m256[size];
		}
		matTransM2();
		for (int i = 0; i < _size; i++)
		{
			for (int j = 0; j < size; j++)
			{
				_mm2[i][j] = _mm256_loadu_ps(&_m2[i][j * 8]);
			}
		}
	}

	~MatMultSIMD()
	{
		for (int i = 0; i < _size; i++) {
			delete[] _mm1[i];
		}
		delete[] _mm1;
		for (int i = 0; i < _size; i++) {
			delete[] _mm2[i];
		}
		delete[] _mm2;
	}

	void matMult()
	{
		int size = _size / 8;
		auto startT = std::chrono::steady_clock::now();
		for (int i = 0; i < _size; i++)
		{
			for (int j = 0; j < _size; j++)
			{
				__m256 t = _mm256_set1_ps(0);
				for (int k = 0; k < size; k++)
				{
					t = _mm256_add_ps(t, _mm256_mul_ps(_mm1[i][k], _mm2[j][k]));
				}
				float sum = 0;
				float* ptr = (float*)(&t);
				for (int pos = 0; pos < 8; pos++)
					sum += ptr[pos];
				_r[i][j] = sum;
			}
		}
		auto endT = std::chrono::steady_clock::now();
		std::chrono::duration<double, std::micro> elapsed = endT - startT;
		cout << right << setw(5)
			<< '1' << '\t' << _size
			<< setw(11) << fixed << setprecision(3)
			<< (double)elapsed.count() / 1000 << "ms" << endl;
	}
};

class MatMultBest :public MatMultSIMD
{
public:
	using MatMultSIMD::MatMultSIMD;
	void matMult(int start, int cnt)
	{
		int size = _size / 8;
		for (int i = start; i < start + cnt; i++)
		{
			for (int j = 0; j < _size; j++)
			{
				__m256 t = _mm256_set1_ps(0);
				for (int k = 0; k < size; k++)
				{
					t = _mm256_add_ps(t, _mm256_mul_ps(_mm1[i][k], _mm2[j][k]));
				}
				float sum = 0;
				float* ptr = (float*)(&t);
				for (int pos = 0; pos < 8; pos++)
					sum += ptr[pos];
				_r[i][j] = sum;
			}
		}
	}
	void matMultTh()
	{
		auto f = [&](int start, int cnt) {matMult(start, cnt); };
		vector<thread> th;
		for (int i = 1; i <= 8; i++)
		{
			int step = _size / i;
			resetR();
			vector<thread> th;
			auto startT = std::chrono::steady_clock::now();
			for (int j = 0; j < i; j++)
			{
				if (j == i - 1)
					th.push_back(thread(f, j * step, _size - j * step));
				else
					th.push_back(thread(f, j * step, step));
			}
			for (auto& i : th)
				i.join();
			auto endT = std::chrono::steady_clock::now();
			std::chrono::duration<double, std::micro> elapsed = endT - startT;
			cout << right << setw(5)
				<< i << '\t' << _size
				<< setw(11) << fixed << setprecision(3)
				<< (double)elapsed.count() / 1000 << "ms" << endl;
		}
	}
};



int main()
{
	/*cout << "common algorithm:" << endl << endl;
	for (int i = 256; i <= 2048; i += 256)
	{
		MatMultCommon* s = new MatMultCommon(i, 1.0);
		s->matMult();
		delete s;
	}
	cout << endl;

	cout << "best algorithm:" << endl << endl;
	for (int i = 256; i <= 2048; i += 256)
	{
		MatMultAlg* s = new MatMultAlg(i, 1.0);
		s->matMult();
		delete s;
	}
	cout << endl;

	cout << "multi-threads:" << endl << endl;
	for (int i = 256; i <= 2048; i += 256)
	{
		MatMultThreads* s = new MatMultThreads(i, 1.0);
		s->matMultTh();
		delete s;
	}
	cout << endl;

	cout << "only SIMD:" << endl << endl;
	for (int i = 256; i <= 2048; i += 256)
	{
		MatMultSIMD* s = new MatMultSIMD(i, 1.0);
		s->matMult();
		delete s;
	}
	cout << endl;*/

	cout << "best performance:" << endl << endl;
	for (int i = 256; i <= 2048; i += 256)
	{
		MatMultBest* s = new MatMultBest(i, 1.0);
		s->matMultTh();
		s->matShowR();
		delete s;
	}
	cout << endl;
}