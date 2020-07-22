double *halton(int i, int m);
double *halton_base(int i, int m, int b[]);
int halton_inverse(double r[], int m);
double *halton_sequence(int i1, int i2, int m);
int i4vec_sum(int n, int a[]);
int prime(int n);
double r8_mod(double x, double y);
void r8mat_print(int m, int n, double a[], std::string title);
void r8mat_print_some(int m, int n, double a[], int ilo, int jlo, int ihi,
                      int jhi, std::string title);
void timestamp();

