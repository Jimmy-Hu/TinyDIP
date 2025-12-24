#ifndef TINYDIP_LINEAR_ALGEBRA_H
#define TINYDIP_LINEAR_ALGEBRA_H

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace TinyDIP
{
namespace linalg
{
    /**
     * @brief A basic Matrix class for linear algebra operations.
     */
    template<typename T>
    class Matrix
    {
    public:
        Matrix() : rows_(0), cols_(0) {}

        Matrix(const std::size_t rows, const std::size_t cols)
            : rows_(rows), cols_(cols), data_(rows * cols)
        {
        }

        T& at(const std::size_t row, const std::size_t col)
        {
            return data_[row * cols_ + col];
        }

        const T& at(const std::size_t row, const std::size_t col) const
        {
            return data_[row * cols_ + col];
        }

        std::size_t rows() const
        {
            return rows_;
        }
        
        std::size_t cols() const
        {
            return cols_;
        }

        bool empty() const
        {
            return rows_ == 0 || cols_ == 0;
        }

        // Other utility functions can be added here (e.g., print)

    private:
        std::size_t rows_;
        std::size_t cols_;
        std::vector<T> data_;
    };

    /**
     * operator<< template function implementation
     * @brief Overloads the stream insertion operator for the Matrix class for easy printing.
     */
    template <typename T>
    std::ostream& operator<<(std::ostream& os, const Matrix<T>& M)
    {
        for (std::size_t r = 0; r < M.rows(); ++r)
        {
            for (std::size_t c = 0; c < M.cols(); ++c)
            {
                os << M.at(r, c) << '\t';
            }
            os << '\n';
        }
        return os;
    }

    /**
     * @brief Transposes a given matrix.
     */
    template<typename ElementT>
    requires (std::floating_point<ElementT>)
    Matrix<ElementT> transpose(const Matrix<ElementT>& A)
    {
        Matrix<ElementT> A_T(A.cols(), A.rows());
        for (std::size_t r = 0; r < A.rows(); ++r)
        {
            for (std::size_t c = 0; c < A.cols(); ++c)
            {
                A_T.at(c, r) = A.at(r, c);
            }
        }
        return A_T;
    }
    
    /**
     * @brief Multiplies two matrices.
     */
    template<typename ElementT>
    requires (std::floating_point<ElementT>)
    Matrix<ElementT> multiply(const Matrix<ElementT>& A, const Matrix<ElementT>& B)
    {
        if (A.cols() != B.rows())
        {
            throw std::runtime_error("Matrix dimensions are incompatible for multiplication.");
        }
        Matrix<ElementT> C(A.rows(), B.cols());
        for (std::size_t i = 0; i < A.rows(); ++i)
        {
            for (std::size_t j = 0; j < B.cols(); ++j)
            {
                ElementT sum = 0.0;
                for (std::size_t k = 0; k < A.cols(); ++k)
                {
                    sum += A.at(i, k) * B.at(k, j);
                }
                C.at(i, j) = sum;
            }
        }
        return C;
    }

    /**
     * @brief Finds eigenvalues and eigenvectors of a real symmetric matrix using the Jacobi eigenvalue algorithm.
     * @param A The input symmetric matrix.
     * @param eigenvalues A vector to be filled with the eigenvalues.
     * @param eigenvectors A matrix whose columns will be the eigenvectors.
     * @param max_iterations Maximum number of sweeps to perform.
     * @param tolerance Convergence tolerance.
     */
    template<typename ElementT>
    requires (std::floating_point<ElementT>)
    void jacobi_eigen_solver(
        const Matrix<ElementT>& A,
        std::vector<ElementT>& eigenvalues,
        Matrix<ElementT>& eigenvectors,
        int max_iterations = 100,
        ElementT tolerance = 1.0e-9)
    {
        if (A.rows() != A.cols())
        {
            throw std::runtime_error("Jacobi solver requires a square matrix.");
        }
        const std::size_t n = A.rows();
        Matrix<ElementT> D = A; // Make a copy to modify

        // Initialize eigenvectors as the identity matrix
        eigenvectors = Matrix<ElementT>(n, n);
        for(std::size_t i = 0; i < n; ++i)
        {
            eigenvectors.at(i, i) = 1.0;
        }
        
        for (int iter = 0; iter < max_iterations; ++iter)
        {
            // Find the largest off-diagonal element
            ElementT max_val = 0.0;
            std::size_t p = 0, q = 1;
            for (std::size_t i = 0; i < n; ++i)
            {
                for (std::size_t j = i + 1; j < n; ++j)
                {
                    if (std::abs(D.at(i, j)) > max_val)
                    {
                        max_val = std::abs(D.at(i, j));
                        p = i;
                        q = j;
                    }
                }
            }

            if (max_val < tolerance) break; // Convergence check

            // Perform Jacobi rotation
            ElementT app = D.at(p, p);
            ElementT aqq = D.at(q, q);
            ElementT apq = D.at(p, q);
            ElementT theta = 0.5 * std::atan2(2 * apq, aqq - app);
            ElementT c = std::cos(theta);
            ElementT s = std::sin(theta);

            // Update D (the matrix being diagonalized)
            D.at(p, p) = c * c * app + s * s * aqq - 2 * s * c * apq;
            D.at(q, q) = s * s * app + c * c * aqq + 2 * s * c * apq;
            D.at(p, q) = D.at(q, p) = 0.0;

            for (std::size_t i = 0; i < n; ++i)
            {
                if (i != p && i != q)
                {
                    ElementT aip = D.at(i, p);
                    ElementT aiq = D.at(i, q);
                    D.at(i, p) = D.at(p, i) = c * aip - s * aiq;
                    D.at(i, q) = D.at(q, i) = s * aip + c * aiq;
                }
            }
            
            // Update eigenvectors matrix
            for(std::size_t i = 0; i < n; ++i)
            {
                ElementT e_ip = eigenvectors.at(i, p);
                ElementT e_iq = eigenvectors.at(i, q);
                eigenvectors.at(i, p) = c * e_ip - s * e_iq;
                eigenvectors.at(i, q) = s * e_ip + c * e_iq;
            }
        }
        
        // Extract eigenvalues from the diagonal of D
        eigenvalues.resize(n);
        for(std::size_t i = 0; i < n; ++i)
        {
            eigenvalues[i] = D.at(i, i);
        }
    }

    /**
     * @brief Solves the system Ah=0 using SVD, by finding the eigenvector of A^T*A with the smallest eigenvalue.
     * @param A The input matrix.
     * @return The vector h that minimizes ||Ah||, which is the last column of V in A=UDV^T.
     */
    template<typename ElementT>
    requires (std::floating_point<ElementT>)
    std::vector<ElementT> svd_solve_ah_zero(const Matrix<ElementT>& A)
    {
        // Form the symmetric matrix A^T * A
        Matrix<ElementT> A_T = transpose(A);
        Matrix<ElementT> ATA = multiply(A_T, A);

        // Find eigenvalues and eigenvectors of A^T * A
        std::vector<ElementT> eigenvalues;
        Matrix<ElementT> eigenvectors;
        jacobi_eigen_solver(ATA, eigenvalues, eigenvectors);

        // Find the index of the smallest eigenvalue
        auto min_it = std::min_element(std::begin(eigenvalues), std::end(eigenvalues));
        std::size_t min_idx = std::distance(std::begin(eigenvalues), min_it);

        // The solution h is the eigenvector corresponding to the smallest eigenvalue
        std::vector<ElementT> h(A.cols());
        for (std::size_t i = 0; i < A.cols(); ++i)
        {
            h[i] = eigenvectors.at(i, min_idx);
        }
        return h;
    }

    /**
     * @brief Computes the inverse of a 3x3 matrix.
     * @return The inverted matrix, or an empty matrix if inversion fails.
     */
    template<typename T>
    Matrix<T> invert(const Matrix<T>& M)
    {
        if (M.rows() != 3 || M.cols() != 3)
        {
            throw std::runtime_error("Matrix inversion is implemented for 3x3 matrices only.");
        }

        T det = M.at(0, 0) * (M.at(1, 1) * M.at(2, 2) - M.at(2, 1) * M.at(1, 2)) -
                M.at(0, 1) * (M.at(1, 0) * M.at(2, 2) - M.at(1, 2) * M.at(2, 0)) +
                M.at(0, 2) * (M.at(1, 0) * M.at(2, 1) - M.at(1, 1) * M.at(2, 0));

        if (std::abs(det) < 1e-9)
        {
            // Matrix is singular and cannot be inverted.
            return Matrix<T>();
        }

        T inv_det = 1.0 / det;
        Matrix<T> inverse(3, 3);

        inverse.at(0, 0) = (M.at(1, 1) * M.at(2, 2) - M.at(2, 1) * M.at(1, 2)) * inv_det;
        inverse.at(0, 1) = (M.at(0, 2) * M.at(2, 1) - M.at(0, 1) * M.at(2, 2)) * inv_det;
        inverse.at(0, 2) = (M.at(0, 1) * M.at(1, 2) - M.at(0, 2) * M.at(1, 1)) * inv_det;
        inverse.at(1, 0) = (M.at(1, 2) * M.at(2, 0) - M.at(1, 0) * M.at(2, 2)) * inv_det;
        inverse.at(1, 1) = (M.at(0, 0) * M.at(2, 2) - M.at(0, 2) * M.at(2, 0)) * inv_det;
        inverse.at(1, 2) = (M.at(1, 0) * M.at(0, 2) - M.at(0, 0) * M.at(1, 2)) * inv_det;
        inverse.at(2, 0) = (M.at(1, 0) * M.at(2, 1) - M.at(2, 0) * M.at(1, 1)) * inv_det;
        inverse.at(2, 1) = (M.at(2, 0) * M.at(0, 1) - M.at(0, 0) * M.at(2, 1)) * inv_det;
        inverse.at(2, 2) = (M.at(0, 0) * M.at(1, 1) - M.at(1, 0) * M.at(0, 1)) * inv_det;

        return inverse;
    }

    /**
     * is_symmetry template function implementation
     * @brief Checks if a matrix is symmetric.
     */
    template<typename T>
    constexpr bool is_symmetry(const Matrix<T>& mat)
    {
        if (mat.rows() != mat.cols())
        {
            return false;
        }

        // For floating point comparisons
        constexpr double epsilon = 1e-6;

        for (std::size_t r = 0; r < mat.rows(); ++r)
        {
            for (std::size_t c = r + 1; c < mat.cols(); ++c)
            {
                const auto val1 = mat.at(r, c);
                const auto val2 = mat.at(c, r);

                if constexpr (std::is_integral_v<T>)
                {
                    if (val1 != val2)
                    {
                        return false;
                    }
                }
                else if constexpr (arithmetic<T>)
                {
                    // Works for floating point and std::complex (magnitude of difference)
                    if (std::abs(val1 - val2) > epsilon)
                    {
                        return false;
                    }
                }
                else
                {
                    // Fallback for non-arithmetic types (e.g., enum class, std::optional, std::string)
                    if (val1 != val2)
                    {
                        return false;
                    }
                }
            }
        }
        return true;
    }

} // namespace linalg
} // namespace TinyDIP

#endif // TINYDIP_LINEAR_ALGEBRA_H