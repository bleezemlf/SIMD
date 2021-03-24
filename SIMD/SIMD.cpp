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
	float ** m_m1;
	float ** m_m2;
	int  m_size;
	float ** m_r;
public:
	MatMult(int size , float value)
	{
		m_size = size;
		m_m1 = new float* [m_size];
		for (int i = 0; i < m_size; i++)
		{
			m_m1[i] = new float[m_size];
		}
		for (int i = 0; i < m_size; i++)
		{
			for (int j = 0; j < m_size; j++)
			{
				m_m1[i][j] = value;
			}
		}
		m_m2 = new float* [m_size];
		for (int i = 0; i < m_size; i++)
		{
			m_m2[i] = new float[m_size];
		}
		for (int i = 0; i < m_size; i++)
		{
			for (int j = 0; j < m_size; j++)
			{
				m_m2[i][j] = value;
			}
		}
		m_r = new float* [m_size];
		for (int i = 0; i < m_size; i++)
		{
			m_r[i] = new float[m_size];
		}
		for (int i = 0; i < m_size; i++)
		{
			for (int j = 0; j < m_size; j++)
				m_r[i][j] = 0;
		}
	}

	~MatMult()
	{
		for (int i = 0; i < m_size; i++) {
			delete[] m_m1[i];
		}
		delete[] m_m1;
		for (int i = 0; i < m_size; i++) {
			delete[] m_m2[i];
		}
		delete[] m_m2;
		for (int i = 0; i < m_size; i++) {
			delete[] m_r[i];
		}
		delete[] m_r;
	}

	virtual void matMult() = 0;
	
	void matShowR()
	{
		for (int i = 0; i < m_size; i++)
		{
			for (int j = 0; j < m_size; j++)
			{
				cout << m_r[i][j];
			}
			cout << endl;
		}
	}

	void matTransM2()
	{
		int t;
		for (int i = 0; i < m_size; i++)
		{
			for (int j = 0; j < i; j++)
			{
				t = m_m2[i][j];
				m_m2[i][j] = m_m2[j][i];
				m_m2[j][i] = t;
			}
		}
	}

	void resetR()
	{
		for (int i = 0; i < m_size; i++)
		{
			for (int j = 0; j < m_size; j++)
				m_r[i][j] = 0;
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
		for (int i = 0; i < m_size; i++)
		{
			for (int j = 0; j < m_size; j++)
			{
				for (int n = 0; n < m_size; n++)
				{
					m_r[i][j] += m_m1[i][n] * m_m2[n][j];
				}
			}
		}
		auto endT = std::chrono::steady_clock::now();
		std::chrono::duration<double, std::micro> elapsed = endT - startT;
		cout << "总占用时间:" << right << setw(10) << fixed << setprecision(3) 
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
			for (int j = 0; j < m_size; ++j)
			{
				int temp = 0;
				for (int k = 0; k < m_size; ++k)
					temp += m_m1[i][k] * m_m2[j][k];
				m_r[i][j] = temp;
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
			int step = m_size / i;
			resetR();
			vector<thread> th;
			auto startT = std::chrono::steady_clock::now();
			for (int j = 0; j < i; j++)
			{
				if (j == i - 1)
					th.push_back(thread(f, j * step, m_size - j * step));
				else
					th.push_back(thread(f, j * step, step));
			}
			for (auto& i : th)
				i.join();
			auto endT = std::chrono::steady_clock::now();
			std::chrono::duration<double, std::micro> elapsed = endT - startT;
			cout << "总占用时间:" << right << setw(10) << fixed 
				<< setprecision(3)<< (double)elapsed.count() / 1000 << "ms" << endl;
		}
	}
};

int main()
{
	//MatMultCommon s1(512, 1.0);
	//s1.matMult();
	
	MatMultThreads s2(1024, 1.0);
	s2.matMultTh();
	
}