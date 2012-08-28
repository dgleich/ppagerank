/*
 * David Gleich
 * 12 January 2007
 * Copyright, Stanford University, 2007
 */

#ifndef YASMIC_BVGRAPH_MATRIX_H
#define YASMIC_BVGRAPH_MATRIX_H

/**
 * @file bvgraph_matrix.h
 * Interprete a Boldi-Vigna Graph as a matrix.
 */
 
#include <iostream>
#include <istream>
#include <fstream>
#include <string>
#include <algorithm>
#include <map>
#include <vector>
#include <boost/lexical_cast.hpp>
#include <iterator>

namespace yasmic 
{
    // prototype the internal classes
    namespace impl {
        class bit_istream;
        class bvgraph_sequential_iterator;
    }

    class bvgraph_matrix 
    {
    private:
        // number of nodes
        int n;
        
        // number of arcs
        int m;
        
        // the base filename of the graph
        std::string basename;
        
        // reference counts
        static const int DEFAULT_MAX_REF_COUNT = 3;
        int _max_ref_count;
               
        static const int DEFAULT_WINDOW_SIZE = 7;
        int _window_size;
        
        static const int DEFAULT_MIN_INTERVAL_LENGTH = 3;
        int _min_interval_length;
        
        static const int DEFAULT_ZETA_K = 3;
        int _zeta_k;
      
        void load_internal();
        
        // disable copy construction
        bvgraph_matrix(const bvgraph_matrix&);
        bvgraph_matrix& operator= (const bvgraph_matrix&);
        
    public:
        // constructor
        bvgraph_matrix(const char* filename)
        : n(0), m(0), basename(filename), 
          _max_ref_count(DEFAULT_MAX_REF_COUNT),
          _window_size(DEFAULT_WINDOW_SIZE),
          _min_interval_length(DEFAULT_MIN_INTERVAL_LENGTH),
          _zeta_k(DEFAULT_ZETA_K)
        {
            load_internal(); 
        }
        int num_nodes() const  { return (n); }
        int num_arcs() const { return (m); }
        
        std::string graph_filename() const { return (basename + ".graph"); }
        int max_ref_count() const  { return (_max_ref_count); }
        int window_size() const { return (_window_size); }
        int min_interval_length() const { return (_min_interval_length); }
        int zeta_k() const { return (_zeta_k); }
        
        typedef impl::bvgraph_sequential_iterator sequential_iterator;
        
        
    }; // class bvgraph_matrix
    
    namespace impl 
    {
        class bit_istream
        {
        private:
            // Precomputed least significant bits for bytes (-1 for 0 ).
            const static int BYTELSB[];
            // Precomputed most significant bits for bytes (-1 for 0 ).
            static const int BYTEMSB[];
            
        public:
            bit_istream(std::istream& is, const int buffer_size)
            : f(is),read_bits(0),current(0),buffer(NULL),bufsize(buffer_size),
              fill(0),pos(0),avail(0),position(0)
            {
                assert ( bufsize > 0 );
                buffer = new unsigned char[bufsize];
            }
            
            ~bit_istream() { if (buffer) { delete[] buffer; } }
            
            /**
             * Completely reset the internal buffers so that the 
             * underlying istream is positionable.
             */
            void flush() {
                position += pos;
                avail = 0;
                pos = 0;
                fill = 0;
            }
            
            /**
             * Close the bitstream
             */
            void close() {
                if (buffer) { delete[] buffer; }
                buffer = NULL;
            }
            
            int read_bit()
            {
                return read_from_current(1); 
            }
            
            /**
             * Read a fixed number of bits into an integer.
             */
            int read_int(int len);          
            int read_unary(); 
            int read_gamma();
            int read_zeta(const int k);
            
            
        private:
            // the underlying file stream
            std::istream &f;
            // the number of bits actually read
            long read_bits;
            // current bit buffer, the lowest fill bits represent the current contents
            int current;
            // the stream buffer
            unsigned char *buffer;
            // the buffer size
            unsigned int bufsize;
            // current number of bits in the bit buffer (stored low)
            int fill;
            // current position in the byte buffer
            int pos;
            // current number of of bytes available in the byte buffer
            int avail;
            // current position of the first byte in the byte buffer
            long position;
            
            /**
             * Read the next byte from the underlying stream.
             * 
             * This method does not update read_bits.
             * 
             * @return the byte
             */
            int read() 
            {
                if (avail == 0) 
                {
                    f.read((char*)buffer, bufsize);
                    avail = f.gcount(); 
                    if (avail <= 0) {
                        // throw an exception
                    }
                    else {
                        position += pos;
                        pos = 0;
                    }
                }
                
                avail--;
                return buffer[pos++] & 0xFF;
            }
            
            /** 
             * Fills {@link #current} to 16 bits.
             */
            int refill16() 
            {
                assert ( fill >= 8 );
                assert ( fill < 16 );
                
                if (avail > 0) {
                    // if there is a current byte in the buffer, use it directly.
                    avail--;
                    current = (current << 8) | (buffer[pos++] & 0xFF);
                    return (fill += 8);
                }
                
                current = (current << 8) | read();
                return (fill += 8);
            }
            
            int refill() 
            {
                if (fill == 0) {
                    current = read();
                    return (fill = 8);
                }
                
                if (avail > 0) {
                    avail--;
                    current = (current << 8) | (buffer[pos++] & 0xFF);
                    return (fill += 8);
                }
                
                current = (current << 8) | read();
                return (fill += 8);
            }
            
            /**
             * Read bits from the buffer, possibly refilling it.
             */
            int read_from_current(const int len) 
            {
                if (len == 0) { return 0; }
                if (fill == 0) { current = read(); fill = 8; }
                read_bits += len;
                unsigned int rval = (unsigned)current;
                return (rval >> (fill -= len) & ((1 << len) - 1));
            }
        }; // class bit_istream
        
        class bvgraph_sequential_iterator
        {
        private:
            // the graph size
            int n;
            
            // the underlying stream
            std::ifstream graph_stream;
            bit_istream bis;
            
            int max_ref_count;
            int window_size;
            int min_interval_length;
            int zeta_k;
            
            // variables for the internal iterators
            bool _row_arcs_end;
            bool _rows_end; 
            int cyclic_buffer_size;
            std::vector<int> outd;
            int curr;
            int curr_outd;
            int curr_arc;
            std::vector<std::vector<int> > window;
            
            std::vector<int> buffer1;
            std::vector<int> buffer2;
            std::vector<int> arcs;
            
            int read_offset() { return (bis.read_gamma()); }    
            int read_outdegree() { return (bis.read_gamma()); }
            int read_reference() { return (bis.read_unary()); }
            int read_block() { return (bis.read_gamma()); }
            int read_block_count() { return (bis.read_gamma()); }
            int read_residual() { return (bis.read_zeta(zeta_k)); }
            
            int nat2int(const int x) { return x % 2 == 0 ? x >> 1 : -( ( x + 1 ) >> 1 ); }
            
            void load_successors();
            
        public:
            // constructor
            bvgraph_sequential_iterator(
                const bvgraph_matrix& m)
            : n(m.num_nodes()),
              graph_stream(m.graph_filename().c_str(),std::ios::binary),
              bis(graph_stream, 16*1024),
              max_ref_count(m.max_ref_count()),
              window_size(m.window_size()),
              min_interval_length(m.min_interval_length()),
              zeta_k(m.zeta_k()),
              _row_arcs_end(false),
              _rows_end(false),
              cyclic_buffer_size(window_size+1),
              outd(cyclic_buffer_size),
              curr(-1),
              window(cyclic_buffer_size)                
            {}
                    
            
            // reset the iterators associated with this graph,
            // the row pointer is set to the first row
            void reset()
            {
                // reset all the file pointers
                graph_stream.seekg(0, std::ios_base::beg);
                bis.flush();
                
                _row_arcs_end = false;
                _rows_end = false;
                curr = -1;
            }
            
            // step to the next row of the matrix
            void next_row()
            {
                // check if we are done
                curr++;
                if (curr >= n-1) { _rows_end = true;  }
                
                int curr_index = curr % cyclic_buffer_size;
                load_successors();
                curr_arc = -1;
                _row_arcs_end = false;
                if (window[curr_index].size() < (unsigned)outd[curr_index]) {
                    window[curr_index].resize(outd[curr_index]);
                }
                
                // unwrap the buffered output
                for (int i = 0; i < outd[curr_index]; i++) {
                    window[curr_index][i] = arcs[i];
                } 
                
                // check to make sure there is something to do
                if (curr_outd == 0) { _row_arcs_end = true; }
            }
            // get the index of the current row
            int cur_row() { return (curr); }
            // get the current outdegree
            int cur_row_outdegree() { return (curr_outd); }
            // returns true when there are no more rows
            bool rows_end() { return (_rows_end); }
            
            // step to the next arc for the current row
            void next_row_arc()
            {
                curr_arc++;
                if (curr_arc >= curr_outd - 1) { _row_arcs_end = true; return; }
            }
            // get the index of the target of the current arc
            int cur_row_arc_target() { return (arcs[curr_arc]); }
            // returns true when there are no more arcs for the current row
            bool row_arcs_end() { return (_row_arcs_end); }
        }; // class bvgraph_iterator
    } // namespace impl            
}// namespace yasmic


#endif // YASMIC_BVGRAPH_MATRIX_H


