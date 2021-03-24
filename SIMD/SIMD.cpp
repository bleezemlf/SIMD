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
			for (auto& t : th)
				t.join();
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

class MatMultSIMD2 :public MatMult
{
public:
	using MatMult::MatMult;
	void matMult()
	{
		int size = _size / 8;
		__m256 _mm1, _mm2;
		auto startT = std::chrono::steady_clock::now();
		for (int i = 0; i < _size; i++)
		{
			for (int j = 0; j < _size; j++)
			{
				__m256 t = _mm256_set1_ps(0);
				for (int k = 0; k < size; k++)
				{
					_mm1 = _mm256_loadu_ps(&_m1[i][8 * k]);
					_mm2 = _mm256_loadu_ps(&_m2[j][8 * k]);
					t = _mm256_add_ps(t, _mm256_mul_ps(_mm1, _mm2));
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
		MatMultSIMD::matTransM2();
		int size = MatMultSIMD::_size / 8;
		for (int i = start; i < start + cnt; i++)
		{
			for (int j = 0; j < MatMultSIMD::_size; j++)
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
				MatMultSIMD::_r[i][j] = sum;
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
			for (auto& t : th)
				t.join();
			auto endT = std::chrono::steady_clock::now();
			std::chrono::duration<double, std::micro> elapsed = endT - startT;
			cout << right << setw(5)
				<< i << '\t' << _size
				<< setw(11) << fixed << setprecision(3)
				<< (double)elapsed.count() / 1000 << "ms" << endl;
		}
	}
};

class VecAdd
{
protected:
	float* _a;
	float* _b;
	float* _c;
	int _size;
public:
	VecAdd(int size,float value)
	{
		_size = size;
		_a = new float[size];
		for (int i = 0; i < size; i++)
			_a[i] = value;
		_b = new float[size];
		for (int i = 0; i < size; i++)
			_b[i] = value;
		_c = new float[size];
		for (int i = 0; i < size; i++)
			_c[i] = 0;
	}

	virtual ~VecAdd()
	{
		delete[]_a;
		delete[]_b;
		delete[]_c;
	}

	virtual void vecAdd() = 0;

	void vecShowC()
	{
		for (int i = 0; i < _size; i++)
		{
			cout << _c[i] << '\t';
			if ((i + 1) % 1000 == 0)
				cout << endl;
		}
	}

	void resetC()
	{
		for (int i = 0; i < _size; i++)
			_c[i] = 0;
	}
};

class VecAddCommon:public VecAdd
{
public:
	using VecAdd::VecAdd;
	void vecAdd()
	{
		auto startT = std::chrono::steady_clock::now();
		for (int i = 0; i < _size; i++)
		{
			_c[i] = _a[i] + _b[i];
		}
		auto endT = std::chrono::steady_clock::now();
		std::chrono::duration<double, std::micro> elapsed = endT - startT;
		cout << right << setw(5) << setprecision(0) << scientific
			<< '1' << '\t'  << double(_size)
			<< setw(11) << fixed << setprecision(3)
			<< (double)elapsed.count() / 1000 << "ms" << endl;
	}
};

class VecAddThreads :public VecAdd
{
public:
	using VecAdd::VecAdd;

	void vecAdd() {};

	void vecAdd(int start,int cnt)
	{
		for (int i = start; i < start + cnt; i++)
		{
			_c[i] = _a[i] + _b[i];
		}
	}

	void vecAddTh()
	{
		auto f = [&](int start, int cnt) {vecAdd(start, cnt); };
		vector<thread> th;
		for (int i = 1; i <= 8; i+=1)
		{
			int step = _size / i;
			vector<thread> th;
			auto startT = std::chrono::steady_clock::now();
			for (int j = 0; j < i; j++)
			{
				if (j == i - 1)
					th.push_back(thread(f, j * step, _size - j * step));
				else
					th.push_back(thread(f, j * step, step));
			}
			for (auto& t : th)
				t.join();
			auto endT = std::chrono::steady_clock::now();
			std::chrono::duration<double, std::micro> elapsed = endT - startT;
			cout << right << setw(5) << setprecision(0) << scientific
				<< i << '\t' << double(_size)
				<< setw(11) << fixed << setprecision(3)
				<< (double)elapsed.count() / 1000 << "ms" << endl;
		}
	}
};

class VecAddSIMD :public VecAdd
{
public:
	using VecAdd :: VecAdd;
	void vecAdd()
	{
		__m256 ma, mb, mc;
		auto startT = std::chrono::steady_clock::now();
		for (int i = 0; i < _size; i+=8) 
		{
			mc = _mm256_set1_ps(0);
			ma = _mm256_loadu_ps(_a + i);
			mb = _mm256_loadu_ps(_b + i);
			mc = _mm256_add_ps(ma, mb);
			_mm256_store_ps(_c + i, mc);
		}
		auto endT = std::chrono::steady_clock::now();
		std::chrono::duration<double, std::micro> elapsed = endT - startT;
		cout << right << setw(5) << setprecision(0) << scientific
			<< '1' << '\t' << double(_size)
			<< setw(11) << fixed << setprecision(3)
			<< (double)elapsed.count() / 1000 << "ms" << endl;
	}
};

class VecAddBest :public VecAdd
{
public:
	using VecAdd::VecAdd;
	void vecAdd() {};
	void vecAdd(int start, int cnt)
	{
		__m256 ma, mb, mc;
		for (int i = start; i < start + cnt; i+=8)
		{
			mc = _mm256_set1_ps(0);
			ma = _mm256_loadu_ps(&_a[i]);
			mb = _mm256_loadu_ps(&_b[i]);
			mc = _mm256_add_ps(ma, mb); 
			_mm256_store_ps(&_c[i], mc);
		}
	}

	void vecAddTh()
	{
		auto f = [&](int start, int cnt) {vecAdd(start, cnt); };
		vector<thread> th;
		for (int i = 1; i <= 8; i++)
		{
			int remain = _size % (8 * i);
			int step = (_size - remain) / i;
			vector<thread> th;
			auto startT = std::chrono::steady_clock::now();
			for (int j = 0; j < i; j++)
			{
					th.push_back(thread(f, j * step, step));
			}
			for (auto& t : th)
				t.join();
			for (int j = i * step; j < _size; j++)
				_c[j] = _a[j] + _b[j];
			auto endT = std::chrono::steady_clock::now();
			std::chrono::duration<double, std::micro> elapsed = endT - startT;
			cout << right << setw(5) << setprecision(0) << scientific
				<< i << '\t' << double(_size)
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
	for (int i = 256; i <= 256; i += 256)
	{
		MatMultThreads* s = new MatMultThreads(i, 1.0);
		s->matMultTh();
		delete s;
	}
	cout << endl;

	cout << "only SIMD:" << endl << endl;
	for (int i = 256; i <= 256; i += 256)
	{
		MatMultSIMD* s = new MatMultSIMD(i, 1.0);
		s->matMult();
		delete s;
	}
	cout << endl;

	cout << "best performance:" << endl << endl;
	for (int i = 256; i <= 256; i += 256)
	{
		MatMultBest* s = new MatMultBest(i, 1.0);
		s->matMultTh();
		delete s;
	}
	cout << endl;*/
	
	cout << "common algorithm:" << endl << endl;
	for (int i = 5e7; i <= 3e8; i += 5e7)
	{
		VecAddCommon* s = new VecAddCommon(i, 1.0);
		s->vecAdd();
		delete s;
	}

	cout << "multi-threads:" << endl << endl;
	for (int i = 5e7; i <= 3e8; i += 5e7)
	{
		VecAddThreads* s = new VecAddThreads(i, 1.0);
		s->vecAddTh();
		delete s;
	}

	cout << "SIMD:" << endl << endl;
	for (int i = 5e7; i <= 3e8; i += 5e7)
	{
		VecAddSIMD* s = new VecAddSIMD(i, 1.0);
		s->vecAdd();
		delete s;
	}

	cout << "Best:" << endl << endl;
	for (int i = 5e7; i <= 3e8; i += 5e7)
	{
		VecAddBest* s = new VecAddBest(i, 1.0);
		s->vecAddTh();
		delete s;
	}
}