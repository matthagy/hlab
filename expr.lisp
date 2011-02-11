
(eval-when (:compile-toplevel :load-toplevel :execute)
  (require :cllib-math (translate-logical-pathname "cllib:math"))
  (require :cllib-gnuplot (translate-logical-pathname "cllib:gnuplot")))

(setq cllib:*gnuplot-path* "/usr/bin/gnuplot")

(defparameter goal-expression '(+ (* (/ 2 3) pi (expt r 3)) (* (/ 2 3)
 pi (expt R 3)) (* -1 pi (+ (* (expt R 2) (+ d (* (/ -1 2) (expt d
 -1) (+ (expt d 2) (expt r 2) (* -1 (expt R 2)))))) (* (/ -1
 3) (expt (+ d (* (/ -1 2) (expt d -1) (+ (expt d 2) (expt r 2) (*
 -1 (expt R 2))))) 3)))) (* pi (+ (* (/ 1 24) (expt d
 -3) (expt (+ (expt d 2) (expt r 2) (* -1 (expt R 2))) 3)) (* (/ -1
 2) (expt d -1) (expt r 2) (+ (expt d 2) (expt r 2) (* -1 (expt R
 2))))))) )

;; (defparameter goal-expression
;; '(sum (i 1 N)
;;   (+ 
;;    (sum (j 
;; 	 (/ i 4) 
;;          N)
;;     (- j 1))
;;    (sum (j
;; 	 (/ i 2)
;; 	 N)
;;     (- j i N)))))

'( ;;Language Def
;; atoms and quoted expressions are compared by equal
;; symbols are compared by equal with runtime value in lexical scope

;; matches any expr
(&anything)
&anything => (&anything)

;; create lexical binding of expr to symbol 
;; when pattern succeeds
;; when binding exists check if current expr is #'equal to
;; existing bindings
(&bind symbol pattern)
(&bind symbol) => (&bind symbol &anything)
($symbol pattern) => (&bind symbol pattern)
($symbol) => (&bind symbol)
$symbol => ($symbol)

;; destructure matching
;; positional matches
;; rest conses bindings in pattern
(&destruct (positional-patterns)* ((&rest pattern?) or
                                   (&rest+ pattern?))?)
(+ ETC) -> (&destruct '+ ETC)
(+ $a $b) -> (&destruct '+ $a $b)
(sum (&destruct $i $lower $upper) ($summand &constant)) ->
  (&destruct 'sum (&destruct $i $lower $upper) 
                  ($summand &constant))



;; matches a constant expression
(&constant)
&constant => (&constant)

;; reverse match logic (pattern bindings are niled)
(&not pattern)

;; intersection (left to right pattern layering)
(&and {patterns}*)

;; evaluate lisp-expr at runtime with lexical scope created by
;; bindings and compare by equal
(&lisp var lisp-expr)
(&has-subexpr sub-expr) => (&lisp :gensym (has-subexpr sub-expr :gensym))
(&without-subexpr sub-expr) => 
    (&lisp :gensym (not (has-subexpr sub-expr :gensym)))
)

;; Utility Macros
(eval-when (:load-toplevel :compile-toplevel :execute)
  (defmacro with-gensyms (syms &body body)
    `(let ,(mapcar (lambda (def)
		     (list (if (listp def)
			       (car def)
			       def)
			   `(gensym ,(string (if (listp def)
						 (cadr def) def)))))
		   syms)
       ,@body))

  (defmacro sqr (x)
    (if (atom x)
	`(* ,x ,x)
	(with-gensyms (%sqr-op)
	  `(let ((,%sqr-op ,x))
	     (* ,%sqr-op ,%sqr-op))))))



;; AST
(eval-when (:compile-toplevel :load-toplevel :execute)
  (defparameter anything 'anything)


(deftype pattern-type ()
  `(or pattern-base (eql ,anything)))

(defun pattern-list-p (list)
  (and (listp list)
       (loop for el in list
             always (typep el 'pattern-type))))

(deftype pattern-list-type ()
  '(satisfies pattern-list-p))

(defclass pattern-base () ())


(defclass subpattern-base (pattern-base)  
  ((sub-pattern                
     :type pattern-type
     :initarg :sub-pattern 
     :initform anything
     :accessor sub-pattern)))

(defclass binding (subpattern-base)
  ((symbol :type symbol 
          :initarg :symbol
          :initform (error "Must supply symbol")
	  :accessor binding-symbol)
   (annotion :type (or null (member create match))
             :initarg :annotion
             :initform nil
             :accessor binding-annotion)))


(defclass destruct (pattern-base)
  ((positional :type pattern-list-type
               :initarg :positional
               :initform nil
               :accessor destruct-positional)
   (communative :type pattern-list-type
                :initarg :communative
                :initform nil
                :accessor destruct-communative)
   (rest-pattern :type pattern-type
		 :initarg :rest-pattern
                 :initform anything
                 :accessor destruct-rest-pattern)
   (rest-style :type (or null (member + *))
	       :initarg :rest-style
               :initform nil
               :accessor destruct-rest-style)))

(defclass ast-not (subpattern-base) ())

(defclass ast-and (pattern-base)
  ((patterns :type pattern-list-type
               :initarg :patterns
               :initform nil
               :accessor ast-and-patterns)))

(defclass ast-or (pattern-base)
  ((patterns :type pattern-list-type
             :initarg :patterns
             :initform nil
             :accessor ast-or-patterns)))

(defclass ast-lisp (pattern-base)
  ((var :type symbol
       :initarg :var
       :initform (error "Must supply var")
       :accessor ast-lisp-var)
   (expr :initarg :expr
        :initform (error "Must supply expr")
	:accessor ast-lisp-expr))))


;; Parsing
(eval-when (:execute :compile-toplevel :load-toplevel)
;; Macros and associated function used to write parsing mini-language

(defmacro parse-dispatch (expr conds patterns &body default)
  "Fucked up pattern matching mini language to simply parsing
Internals are really fucked up, but it work"
  (with-gensyms (%expr %is-list)
    (let ((inner `(progn ,@default)))
      (dolist (p (reverse patterns))
	(setq inner (write-pd-pattern %expr %is-list p inner)))
      (setq inner `(let ((,%is-list (listp ,%expr)))
		     ,inner))
      (dolist (c (reverse conds))
	(setq inner (write-pd-cond c inner)))
      `(let ((,%expr ,expr))
	 ,inner))))

(defun write-pd-pattern (%expr %is-list pattern-def else-code)
  (declare (optimize (debug 3)
		     (speed 0)
		     (safety 3)))
  (destructuring-bind (pattern &rest body) pattern-def
    (if (or (atom pattern)
	    (eq (car pattern)
		'quote))
	`(if (equal ,pattern ,%expr)
	     (progn ,@body)
	     ,else-code)
	(let (positional rest rest-var)
	  (dolist (el pattern)
	    (cond (rest-var
		   (error "extra shit after &rest var ~A" el))
		  (rest
		   (unless (symbolp el)
		     (error "bad restvar ~A" el))
		   (setq rest-var el))
		  ((eq el '&rest)
		   (setq rest t))
		  (t (push el positional))))
	  (setq positional (nreverse positional))
	  (let* ((pos-length (list-length positional))
		 (dest-check  (make-destruct-length-check %expr pos-length rest))
		 (pos-gensyms (loop repeat pos-length collect (gensym "POS")))
		 (ass-pos (make-destruct-gensym-assigns %expr pos-gensyms)))
	    (multiple-value-bind (checks let-bindings)
		(make-destructuring-checks-and-binds positional pos-gensyms)
	      (when rest
		(push `(,rest-var (nthcdr ,pos-length ,%expr))
		      let-bindings))
	      `(let ,pos-gensyms
		 (if (and ,%is-list ,dest-check
			  (progn ,ass-pos t)
			  ,@checks)
		     (let ,let-bindings ,@body)
		     ,else-code))))))))

(defun make-destruct-gensym-assigns (%expr pos-gensyms)
  (with-gensyms (%acc)
    `(let ((,%acc ,%expr))
       ,@(loop with last = (car (last pos-gensyms))
	    for g in pos-gensyms
	    collect `(setq ,g (car ,%acc))
	    unless (eq g last)
	    collect `(setq ,%acc (cdr ,%acc))))))

(defun make-destruct-length-check (%expr pos-length rest)
  (if rest
      `(nthcdr ,(1- pos-length) ,%expr)
      (with-gensyms (%cons)
	`(let ((,%cons (nthcdr ,(1- pos-length)
			       ,%expr)))
	   (and ,%cons (null (cdr ,%cons)))))))

(defun binding-symbol-p (sym)
  (and (symbolp sym)
       (char= (aref (symbol-name sym)
		    0)
	      #\$)))
 
(defun quotep (op)
  (and (consp op)
       (eq (car op)
	   'quote)))

(defun debind-symbol (sym)
  (intern (subseq (symbol-name sym)
		  1)))

(defun make-destructuring-checks-and-binds (positional gensyms)
  (loop with checks and let-bindings
     for pos in positional and
     gsym in gensyms
     when (binding-symbol-p pos) do
     (push `(binding-symbol-p ,gsym) checks)
     end
     when (quotep pos) do
     (push `(equal ,pos ,gsym) checks)
     else do
     (push (if (binding-symbol-p pos)
	       (list
		(debind-symbol pos)
		`(debind-symbol ,gsym))
	       (list pos gsym))
	   let-bindings)
     end
     finally (return (values checks let-bindings))))

  (defun write-pd-cond (conddef else-code)
    (destructuring-bind (test &rest body) conddef
      `(if ,test 
	   (progn ,@body)
	   ,else-code)))



(defun parse (dsl)
  "parse pattern matching mini-language into ast"
  (declare (optimize (debug 3)
		     (speed 0)
		     (safety 3)
                     (compilation-speed 0)))
  (labels 
    ((make-bind (symbol pattern)
       (make-instance 'binding 
	:symbol symbol 
	:sub-pattern pattern))
     (make-not (pattern)
       (make-instance 'ast-not
	:sub-pattern pattern))
     (make-and (patterns)
       (make-instance 'ast-and
	:patterns patterns))
     (make-or (patterns)
       (make-instance 'ast-or
	:patterns patterns))
     (make-lisp (var expr)
       (make-instance 'ast-lisp
	:var var :expr expr))
     (make-constant ()
        (make-lisp 'expr
             '(typep expr 'number)))
     (make-destruct (positional communative rest-pattern rest-style)
       (make-instance 'destruct 
	:positional positional
        :communative communative
	:rest-pattern rest-pattern
	:rest-style rest-style))
     (parse-destruct (exprs)
       (loop with positional and
                  commuting and
                  communative and
	          rest-pattern and
	          rest-style
	  for expr in exprs do
	  (cond ((member expr '(&rest &rest+))
		 (when rest-style
		   (error "multiple rest specifiers in &destruct"))
		 (setq rest-style (if (eq expr '&rest) '* '+)))
		(rest-style
		 (when rest-pattern
		   (error "multiple rest pattern in &destruct"))
		 (setq rest-pattern (parse expr)))
                ((eql expr '&commute)
                 (when commuting
                   (error "multiple &commute specifiers in &destruct"))
                 (setq commuting t))
                (commuting (push (parse expr) communative))
		(t (push (parse expr) positional)))
	  finally (when (and commuting (null communative))
		    (error "&commute with no patterns"))
                  (return (make-destruct (nreverse positional) 
                                         (nreverse communative)
                                         (or rest-pattern anything)
                                         rest-style)))))
  (parse-dispatch dsl
    (((binding-symbol-p dsl)
      (make-bind (intern (subseq (symbol-name dsl)
				 1))
		 anything))
     ((and (consp dsl) 
           (eq (car dsl) 'quote))
	(make-lisp 'expr `(equal ,dsl expr))))
    (('&anything   anything)
     ('(&anything) anything)
     ('&constant   (make-constant))
     ('(&constant) (make-constant))
     (('&bind symbol pattern)   (make-bind symbol (parse pattern)))
     (('&bind symbol)           (make-bind symbol anything))
     (($symbol pattern)         (make-bind symbol (parse pattern)))
     (($symbol)                 (make-bind symbol anything))
     (('&not pattern) (make-not (parse pattern)))
     (('&and &rest patterns) (make-and (mapcar #'parse patterns)))
     (('&or &rest patterns) (make-or (mapcar #'parse patterns)))
     (('&lisp var expr) (make-lisp var expr))
     (('&has-subexpr sub) (make-lisp 'expr `(has-subexpr ,sub expr)))
     (('&without-subexpr sub) (make-lisp 'expr `(not (has-subexpr ,sub expr))))
     (('&destruct &rest components) (parse-destruct components))
     ((symbol &rest components) (parse-destruct (cons symbol
						      components))))
    (if (atom dsl)
	  (make-lisp 'expr (if (symbolp dsl)
			      `(eql ',dsl expr)
                              `(eql ,dsl expr)))
	(error "invalid expression ~A" dsl))))) 


(defgeneric ast2dsl (node)
  (:documentation "convert ast node to canonical dsl representation"))

(defmethod ast2dsl ((anything (eql anything)))
  '(&anything))

(defmethod ast2dsl ((b binding))
  `(&bind ,(binding-symbol b)
          ,(ast2dsl (sub-pattern b))))

(defmethod ast2dsl ((l ast-lisp))
  `(&lisp ,(ast-lisp-var l)
           ,(ast-lisp-expr l)))

(defmethod ast2dsl ((d destruct))
  (with-slots (positional communative rest-pattern rest-style) d
    (append '(&destruct) 
	    (mapcar #'ast2dsl positional)
            (when communative
              (cons '&commute
                (mapcar #'ast2dsl communative)))
	    (when rest-style
	      (list (if (eq rest-style '+)
			'&rest+ '&rest)
		    (ast2dsl rest-pattern))))))

(defmethod ast2dsl ((n ast-not))
  `(&not ,(ast2dsl (sub-pattern n))))

(defmethod ast2dsl ((a ast-and))
  (append '(&and) 
	  (mapcar #'ast2dsl (ast-and-patterns a))))

(defmethod ast2dsl ((a ast-or))
  (append '(&or) 
	  (mapcar #'ast2dsl (ast-or-patterns a))))

;; Pattern Compiler
(defgeneric ast-children (node)
  (:documentation "list of children in order of execution"))

(defmethod ast-children ((a (eql anything))))

(defmethod ast-children ((s subpattern-base))
  (list (sub-pattern s)))

(defmethod ast-children ((d destruct))
  (append (destruct-positional d)
          (destruct-communative d)
	  (and (destruct-rest-style d)
	       (list (destruct-rest-pattern d)))))

(defmethod ast-children ((a ast-and))
  (copy-list (ast-and-patterns a)))

(defmethod ast-children ((a ast-or))
  (copy-list (ast-or-patterns a)))

(defmethod ast-children ((a ast-lisp)))

(defun ast-walk (walker node)
  "Depth first walk of parse tree"
  (declare (type function walker)
	   (type pattern-type node))
  (funcall walker node)
  (dolist (child (ast-children node))
     (ast-walk walker child)))

(defmacro do-ast-children ((var node) &body body)
  `(ast-walk (lambda (,var) ,@body)
	     ,node))

(defmacro do-ast-children-type ((var node type)
                                 &body body)
  (with-gensyms (%type)
     `(let ((,%type ,type))
	(do-ast-children (,var ,node)
	   (when (typep ,var ,%type)
	     ,@body)))))

(defun collect-bindings (top &optional binding-type)
  (assert (member binding-type '(() create match)))
  (let (acc)
    (do-ast-children-type (node top 'binding)
       (unless (binding-annotion node)
	 (error "un-annotted binding"))
       (when (or (null binding-type) 
                 (eq binding-type  (binding-annotion node)))
	 (push (binding-symbol node)
	       acc)))
    (nreverse (remove-duplicates acc))))

(defun annote-bindings (node bindings)
  (typecase node
    (binding 
     (let ((bound (member (binding-symbol node)
			  bindings :test 'eq)))
       (setf (binding-annotion node)
	     (if bound 'match 'create))
       (if bound 
	   bindings
	   (cons (binding-symbol node)
		 bindings))))
    (ast-not
      (unless (eq (annote-bindings (sub-pattern node) bindings)
		  bindings)
          (error "not expression with created bindings"))
      bindings)
    (ast-or 
     (let ((alternation-bindings 
	    (loop for child in (ast-children node)
	       collect (annote-bindings child bindings))))
       (if alternation-bindings
	 (loop with first-alterantion = (car alternation-bindings)
	    for alt in (cdr alternation-bindings)
	    when (or (set-difference first-alterantion alt)
                     (set-difference alt first-alterantion))
	    do (error "alternations with different bindings created")
	    end
	    finally (return first-alterantion))
         bindings)))
    (t (dolist (child (ast-children node))
	 (setq bindings (annote-bindings child bindings)))
       bindings)))

(defvar *gen-code-let-symbols* nil)
(defvar *gen-code-lables* nil)
(defvar *gen-code-macrolets* nil)

(defun gensym-optional-init (&optional init)
 (if init (gensym (string-upcase (format nil "~a" init))) (gensym)))

(defun create-let-gensym (&optional init init-form)
  (declare (special *gen-code-let-symbols*))
  (let ((g (gensym-optional-init init)))
    (push (if init-form
              (list g init-form)
	      g)
           *gen-code-let-symbols*)
    g))

(defun create-label-name (list base)
  (unless base
     (setq base 'annonymous))
  (loop for i from 0 and
            name = (intern (string-upcase (if (zerop i)
					   (format nil "~a" base)
                                           (format nil "~a~d" base i))))
	    unless (member name list
                     :test #'string= :key #'car) 
              do (return name)))

(defun create-label-ex (basename lambda-var body)
  (declare (special *gen-code-lables*))
  (let ((name (create-label-name *gen-code-lables* basename)))
    (push `(,name (,lambda-var) ,@body)
	  *gen-code-lables*)
    name))

(defun get-label-body (name)
  (loop for lab in *gen-code-lables*
        when (eq (car lab)
		 name)
        do (return-from get-label-body (cddr lab)))
  (error "not found"))

(defmacro create-label ((lambda-var &optional name) &body body)
  `(create-label-ex ,name ',lambda-var (list ,@body)))


(defgeneric code-matcher (ast)
  (:documentation "creates label to match this portion of the ast.
returns label symbol"))

(defun code-pattern-matcher (ast &optional (body-success '(t)) body-failure)
  (unless (typep ast 'pattern-type)
    (setq ast (parse ast)))
  (annote-bindings ast nil)
  (let ((*gen-code-let-symbols* 
	 (collect-bindings ast 'create))
        *gen-code-lables*)
    (declare (special *gen-code-let-symbols* 
                      *gen-code-lables*))
    (let* ((top-label
	    (code-matcher ast)))
      (with-gensyms (%expr)
        `(lambda (,%expr) 
           (declare (optimize (speed 3)
			      (safety 0)
			      (debug 0)
			      (compilation-speed 0)))
	   (let ,*gen-code-let-symbols*
             (labels ,*gen-code-lables*
	       (if (,top-label ,%expr)
		   (progn ,@body-success)
		   (progn ,@body-failure)))))))))

(defun compile-pattern-matcher (ast &optional (body-success '(t))
				              body-failure)
  (compile nil (code-pattern-matcher ast 
                body-success body-failure)))

(defmethod code-matcher ((a (eql anything)))
  (create-label (expr 'anything)
    '(declare (ignore expr))
    t))

(defmethod code-matcher ((b binding))
  (let* ((bsym (binding-symbol b))
	 (sub-matcher (code-matcher (sub-pattern b))))
     (ecase (binding-annotion b)
       (create
	 (create-label 
           (expr (format nil "bind-~a" bsym))
              `(setq ,bsym expr)
              `(,sub-matcher expr)))
       (match
	 (create-label 
           (expr (format nil "match-~a" bsym))
	     `(and (equal ,bsym expr)
                   (,sub-matcher expr)))))))

(defmethod code-matcher ((lisp ast-lisp))
  (create-label-ex 'lisp (ast-lisp-var lisp)
    (list (ast-lisp-expr lisp))))

(defmethod code-matcher ((a ast-and))
  (create-label (expr 'pand)
    `(and ,@(loop for p in (ast-and-patterns a)
		 collect `(,(code-matcher p)
			    expr)))))

(defmethod code-matcher ((a ast-or))
  (create-label (expr 'por)
    `(or ,@(loop for p in (ast-or-patterns a)
		 collect `(,(code-matcher p)
			    expr)))))

(defmethod code-matcher ((a ast-not))
  (create-label (expr 'pnot)
    `(not (,(code-matcher (sub-pattern a)) expr))))

(defmethod code-matcher ((d destruct))
  (with-slots (positional communative rest-pattern rest-style) d
    (create-label (expr 'destruct)
      `(block destruct
	(unless (listp expr)
	  (return-from destruct))
	,@(make-destructor positional (+ (list-length communative)
					 (if (eq rest-style '+)
					     1 0)))
	,(make-commuting-and-rest (list-length communative) communative 
                                  nil (and rest-style rest-pattern))))))

(defun make-nthcdr (n)
  (case n
    (1 'expr)
    (2 '(cdr expr))
    (3 '(cddr expr))
    (4 '(cdddr expr))
    (5 '(cddddr epxr))
    (otherwise `(nthcdr ,(1- n) expr))))

(defun make-destructor (positional min-rest-length)
  (append
    (loop for p in positional 
        collect 
          '(unless expr
	    (return-from destruct))
        collect
          `(unless (,(code-matcher p) (car expr))
	     (return-from destruct))
        collect
          '(setq expr (cdr expr)))
    
    (when (> min-rest-length 0)
      (list      
        `(unless ,(make-nthcdr min-rest-length)
	   (return-from destruct))))))

(defun make-communative-cons-check (communative-cons-syms)
  `(or ,@(loop for cc in communative-cons-syms
			collect `(eq cons ,cc))))

(defun make-commuting-and-rest (n-communative communative 
                                communative-cons-syms 
                                rest-pattern)
  (if communative
     (let ((comm-cons (intern (format nil "COM-CONS~d" 
                                (list-length communative-cons-syms)))))
       `(loop with ,comm-cons
              for cons on expr do
                 (unless ,(make-communative-cons-check communative-cons-syms)
		   (when (,(code-matcher (car communative)) (car cons))
		     (setq ,comm-cons cons)
		     ,(make-commuting-and-rest n-communative
                                               (cdr communative)
					       (cons comm-cons
						     communative-cons-syms)
					          rest-pattern)))))
     (if rest-pattern
        (make-rest-pattern rest-pattern communative-cons-syms)
         `(return-from destruct (null ,(make-nthcdr (1+ n-communative)))))))


(defun make-rest-pattern (rest-pattern communative-cons-syms)
  (when (collect-bindings rest-pattern 'match)
    (error "rest pattern with matching bindings"))
  (let* ((bindings (collect-bindings rest-pattern 'create))
	 (binding-lists-syms (loop for binding in bindings
                                collect (intern (format nil "ACC-~a" binding)))))
    `(let ,binding-lists-syms
       (loop
	  for cons on expr do
	  (unless ,(make-communative-cons-check communative-cons-syms)
	    (if (,(code-matcher rest-pattern) (car cons))
		(progn ,@(loop for binding in bindings and
			    sym in binding-lists-syms
			    collect `(push ,binding ,sym)))
  	        (return)))
	  finally 
	  ,@(loop for binding in bindings and
	       sym in binding-lists-syms
	       collect `(setq ,binding (nreverse ,sym)))
	  (return-from destruct t)))))


)




(progn

(defparameter transformers nil)

(defun register-pattern-transformer (name groups transformer)
  (when (symbolp groups)
    (setq groups (list groups)))
  (push (list name groups transformer)
	transformers))

(eval-when (:execute :compile-toplevel :load-toplevel)
  (defmacro def-pattern-transformer (name groups pattern &body body)
    `(register-pattern-transformer ',name ',groups ,(code-pattern-matcher pattern body))))


;; Extraneous expression removers

(def-pattern-transformer remove-empty-addition remove-extraneous
  (+)
  0)

(def-pattern-transformer remove-empty-subtraction remove-extraneous
  (+)
 n 1)

(def-pattern-transformer remove-empty-multiplication remove-extraneous
  (*)
  1)

(def-pattern-transformer remove-empty-division remove-extraneous
  (/)
  0)

(def-pattern-transformer remove-single-addition remove-extraneous
  (+ $addand)
   addand)

(def-pattern-transformer remove-single-multiplication remove-extraneous
  (* $multiplicand)
   multiplicand)

(def-pattern-transformer remove-zero-multiplication 
  (remove-extraneous constant-folding)
  (* &commute 0 &rest)
  0)

(def-pattern-transformer remove-zero-addition
  (remove-extraneous constant-folding)
  (+ &commute 0 &rest+ $addands)
  `(+ ,@addands))

(def-pattern-transformer remove-one-multiplication 
  (remove-extraneous constant-folding)
  (* &commute 1 &rest+ $rest)
  `(* ,@rest))

(def-pattern-transformer remove-negone-multiplication
  (remove-extraneous constant-folding)
  (* &commute '(- 1) &rest+ $multiplicands)
  `(- (* ,@multiplicands)))

(def-pattern-transformer insert-negone-multiplication expanding
  (- $x)
  `(* -1 ,x))

;; Expression folders
(def-pattern-transformer fold-additon (folding expanding)
  (+ &commute (+ &rest $addands1) &rest $addands2)
  `(+ ,@(append addands1 addands2)))

(def-pattern-transformer fold-multiplication (folding expanding)
  (* &commute (* &rest $multiplicand1) &rest $multiplicand2)
  `(* ,@(append multiplicand1 multiplicand2)))

;; Convert subtraction to addition of negative terms
;; Allows for communative operations on negative
(def-pattern-transformer remove-subtraction expanding
  (- $first &rest+ $rest)
   `(+ ,first ,@(loop for sub in rest
		  collect `(- ,sub))))

;; Reverse above transformation (useful after 
;; communative transformations)
(def-pattern-transformer insert-subtraction contracting
  (+ &commute (- $neg) &rest+ $addands)
  `(- (+ ,@addands)
      ,neg))

(def-pattern-transformer fold-subtraction folding
  (- &commute (+ &commute (- $neg) 
                 &rest+ $addands) 
     &rest+ $negands)
  `(- (+ ,@addands) ,neg ,@negands))

;; Same for division
(def-pattern-transformer remove-division expanding
    (/ $first &rest+ $rest)
  `(* ,first ,@(loop for div in rest
		  collect `(/ ,div))))

(def-pattern-transformer insert-division contracting
  (* &commute (/ $div) &rest+ $multiplicands)
  `(/ (* ,@multiplicands)
      ,div))

(def-pattern-transformer fold-division folding
  (/ &commute (* &commute (/ $div) 
                 &rest+ $multiplicands) 
     &rest+ $divands)
  `(/ (* ,@multiplicands) ,div ,@divands))

;; Constant reduction
(def-pattern-transformer reduce-constant-negative constant-folding
  (- ($c &constant))
  (- c))

(def-pattern-transformer reduce-constant-inverse constant-folding
  (/ ($c &constant))
  (/ c))

(def-pattern-transformer reduce-constant-addition constant-folding
  (+ &commute ($c1 &constant)
              ($c2 &constant)
     &rest $addands)
  `(+ ,(+ c1 c2)
      ,@addands))

(def-pattern-transformer reduce-constant-multiplication constant-folding
  (* &commute ($c1 &constant)
              ($c2 &constant)
     &rest $multiplicands)
  `(* ,(* c1 c2)
      ,@multiplicands))

(def-pattern-transformer reduce-constant-expt constant-folding
  (expt ($base &constant) ($power &constant))
  (expt base power))


(def-pattern-transformer fold-add-to-multiply folding
  (+ &commute $add $add 
     &rest $addands)
  `(+ (* 2 ,add) ,@addands))

(def-pattern-transformer associate-multiplication expanding
  (* &commute (+ &rest+ $addands)
     &rest $multiplicands)
 `(+ ,@(loop for addand in addands
        collect `(* ,addand ,@multiplicands))))

(def-pattern-transformer combine-terms folding
   (&or (+ &commute (* $term1 ($x (&not &constant)))
	   (* &commute $term2 $x)
           &rest $addands)
        (+ &commute (* ($x (&not &constant)) $term1)
	   (* &commute $term2 $x)
	   &rest $addands))
  `(+ (* (+ ,term1 ,term2) ,x) ,@addands))

(def-pattern-transformer add-term  folding
  (+ &commute (* &commute $term $x) $x
     &rest $addands)
  `(+ (* (+ ,term 1) ,x) ,@addands))

;; Exponential expanders
(def-pattern-transformer expand-expt expanding
  (expt $base ($power (&lisp p
                           (and (integerp p)
				(>= p 0)))))
    `(* ,@(loop repeat power collect base)))

(def-pattern-transformer create-expt folding
  (* &commute $base $base &rest $multiplicands)
  `(* (expt ,base 2)
      ,@multiplicands))

(def-pattern-transformer join-expt folding
  (* &commute (expt $base $power) $base
     &rest $multiplicands)
  `(* (expt ,base (+ ,power 1))
      ,@multiplicands))

(def-pattern-transformer fold-expts folding
  (* &commute (expt $base $power1)
              (expt $base $power2)
     &rest $multiplicands)
  `(* (expt ,base (+ ,power1 ,power2))
      ,@multiplicands))

;; sums
(def-pattern-transformer desocciate-sums expanding
    (sum $descriptor (+ &rest+ $addands))
  `(+ ,@(loop for e in addands
	   collect `(sum ,descriptor ,e))))

(def-pattern-transformer convert-multiplier-sums expanding
    (sum (&destruct $i $start $end)
         (* &commute ($constant (&without-subexpr i))
 	    &rest+ $multiplicands))
  `(* ,constant  (sum (,i ,start ,end) (* ,@multiplicands))))

;; Reduce sums to simpler expressions
(def-pattern-transformer convert-constant-sums expanding
    (sum (&destruct $i $start $end)
          ($constant (&without-subexpr i)))
  `(* (+ 1 (- ,end ,start))
      ,constant))

(defmacro def-sum-reducer (name
                           (index-var pattern)
                           flet-def)
  (with-gensyms (%summer %start %end)
    `(def-pattern-transformer ,name expanding
     (sum (&destruct ,index-var
                     (&bind ,%start) 
                     (&bind ,%end))
            ,pattern)
     (flet ((,%summer ,@flet-def))
       (list '- (,%summer ,%end)
	        (,%summer `(- ,,%start 1)))))))

(def-sum-reducer convert-sum-series 
  ($i $i)
  ((expr)
  `(/ (* (+ ,expr 1)
	 ,expr)
      2)))

(def-sum-reducer convert-sum-squares
  ($i (* $i $i))
  ((expr)
   `(/ (* ,expr
	  (+ ,expr 1)
	  (+ (* 2 ,expr)
	     1))
       6)))

(def-sum-reducer convert-sum-cubes
  ($i (* $i $i $i))
  ((expr)
   `(/ (* (expt ,expr 2)
	  (expt (+ ,expr 1) 2))
       4))))


(defun find-transformations (expr)
 (declare (optimize (speed 2)
		    (safety 3)
		    (debug 0)
		    (compilation-speed 0)))
  (let (acc trans)
    (dolist (transdef transformers)
      (setq trans (funcall (the function (cdr transdef))
			   expr))
      (when trans
;	(format t "~s ~a -> ~a~%" (car transdef) expr trans)
         (push trans acc)))
    acc))

(defun get-explosion-and-imploder (expr)
  "Returns a list of subexpressions in expression and
a closure that builds a similar expression with replaced
subexpressions"
 (declare (optimize (speed 2)
		    (safety 3)
		    (debug 0)
		    (compilation-speed 0)))
  (cond ((atom expr)
	 (values (list expr)
		 #'car))
	((eq (car expr) 'sum)
	 (destructuring-bind ((index-sym lower-expr upper-expr)
			      summand-expr) (cdr expr)
	   (values (list lower-expr upper-expr summand-expr)
		   (lambda (replacands)
		     `(sum (,index-sym 
			    ,(car replacands) 
			    ,(cadr replacands))
			   ,(caddr replacands))))))
	(t 
	 (values (cdr expr)
		 (lambda (replacands)
		   (cons (car expr)
			 replacands))))))

;; Could definitely optimize this (through open coded walking) 
;; but likely unncessary now
(defun has-subexpr (subexpr expr)
  (if (atom expr)
      (eql subexpr expr)
      (multiple-value-bind (subexprs imploder)
	  (get-explosion-and-imploder expr)
          (loop for sub in subexprs
                thereis (has-subexpr subexpr sub)))))

(defun multiplex-lists (lists)
"(multiplex-lists '((105 ultegra duraAce) (pussy mouth ass))) =>
 '((DURAACE PUSSY) (DURAACE MOUTH) (DURAACE ASS) 
   (ULTEGRA PUSSY) (ULTEGRA MOUTH) (ULTEGRA ASS)   
   (105 PUSSY)     (105 MOUTH)     (105 ASS))"
 (declare (type list lists)
          (optimize (speed 2)
		    (safety 3)
		    (debug 0)
		    (compilation-speed 0)))
 (assert lists)
 ;; recurse with labels for optimizations
 (labels ((rec (lists)
	    (declare (type list lists))
	    (if (null lists)
                (list nil)
		(let ((subs (rec (cdr lists)))
		      acc)
		  (dolist (el (car lists) acc)
		    (dolist (sub subs)
		      (push (cons el sub)
			    acc)))))))
   (rec lists)))

(defun communative-p (expr)
  (and (listp expr)
       (member (car expr)
	       '(+ *))))

(defun list2array (list)
 (declare (optimize (speed 3)
		    (safety 0)
		    (debug 0)
		    (compilation-speed 0))
	  (type list list))
  (make-array (list-length list)
	      :initial-contents list))

(defun array2list (array)
 (declare (optimize (speed 3)
		    (safety 0)
		    (debug 0)
		    (compilation-speed 0))
	  (type (simple-array * (*))
		array))
   (loop for el across array
        collect el))

(defun permutate-list (list)
 (declare (optimize (speed 3)
		    (safety 0)
		    (debug 0)
		    (compilation-speed 0)))
  (mapcar #'array2list (cllib:permutations-list (list2array list))))

(defun score-expression (expr)
  (cond ((numberp expr) 1)
	((symbolp expr) 100)
        (t (+ (ecase (car expr)
		((+ -) 10)
                ((* /) 25)
                ((exp) 40)
		((expt) 45)
		(sum 10))
	      (loop for sub in (get-explosion-and-imploder expr)
                   sum (score-expression sub))))))

(defparameter temperature 10)

(defun recursive-search-ex (expr)
  (declare (optimize (speed 2)
		     (safety 3)
		     (debug 0)
		     (compilation-speed 0)))
  (let ((rebuild-expr
	 (multiple-value-bind (subexprs imploder) 
	     (get-explosion-and-imploder expr)
	   (funcall (the function imploder)
		    (mapcar #'recursive-search subexprs))))
	weighted-expressions
	weight
	(total-weight 0.0d0)
	(inv-neg-T (* -1.0d0 (/ temperature)))
	choice)
    (dolist (expr (cons rebuild-expr (find-transformations rebuild-expr)))
      (setq weight (exp (* (score-expression expr)
			   inv-neg-T)))
      (push (cons weight expr) weighted-expressions)
      (setq total-weight (+ weight total-weight)))
    (setq choice (* total-weight (random 1.0d0)))    
    ;;        (format t "total ~a~%" total-weight)
    ;;        (format t "choice ~a~%" choice)
    (dolist (cell weighted-expressions)
      (setq choice (- choice (car cell)))
      (when (<= choice 0)
	(return-from recursive-search-ex (cdr cell))))   
    (error "FUCK")))

(defun recursive-search (expr)
  (if (atom expr) expr
    (dotimes (n (floor (score-expression expr) 100))
      (setq expr (recursive-search-ex expr))))
    expr)

(defun annealing-loop (expr &optional (iterations 1000))
  (loop for i below iterations 
        until (atom expr) do
        (when (eq 0 (mod i 1))
        (format t "~a~%" expr))
        (setq expr (recursive-search expr)))
  expr)

(defun plot-annealings (expr temps iterations)
  (declare (special temperature))
  (cllib:with-plot-stream (str :xlabel "Time"
                               :ylabel "Score"
                               :title (format nil "Annealing of ~A" expr))
    (format str "unset nokey~%")
    (format str "set key right top~%")
    (format str "plot")
    (loop for first = t then nil
          for temp in temps do
	 (unless first
	   (format str ","))
	 (format str " '-' u 1:2 title \"T=~A\" with lp" temp))
    (terpri str)
    (dolist (temp temps)
      (let ((temperature temp))
      (loop for iteration upto iterations
            for cexpr = expr then (recursive-search cexpr)
            for score =  (score-expression cexpr)
	    do (when (eq 0 (mod iteration (floor iterations 50)))
		 (format str "~d ~f~%" iteration score))
               (when (eq 0 (mod iteration (floor iterations 10)))
                 (format t "~a ~d ~f~%" temp iteration score))
	    finally (format t "~a~%" cexpr)))
      (format str "e~%"))
    ))

(defun recursive-transform (expr)
  (declare (optimize (speed 2)
		    (safety 3)
		    (debug 0)
		    (compilation-speed 0)))
  (if (atom expr) 
      (list expr)
      (let ((permutated-exprs
             (multiple-value-bind (subexprs imploder) 
		 (get-explosion-and-imploder expr)
	       ;; permutate subexpressions         
	       (mapcar (the function imploder)
		  (let ((permutated-subexprs
			 (remove-duplicates
			  (multiplex-lists 
			   (mapcar #'recursive-transform subexprs))
			  :test #'equal)))
		    (if (member (car expr) '(+ *)
				:test #'eq)
			(remove-duplicates
			 (let (acc)
			   (dolist (subexprs permutated-subexprs)
			     (setq acc (nconc acc (permutate-list subexprs))))
			   acc)
			 :test #'equal)
			permutated-subexprs)))))
	    acc-transformed)
	(dolist (expr permutated-exprs)
	  (setq acc-transformed (cons expr 
             (nconc acc-transformed 
               (find-transformations expr)))))
	acc-transformed)))

(defun find-all-transformations (last-exprs)
 (declare (optimize (speed 2)
		    (safety 3)
		    (debug 0)
		    (compilation-speed 0))
	  (type list last-exprs))
  (let (all)
    (loop for i upto 2
          while last-exprs 
          do
       (setq all (nconc all last-exprs))
       (let (acc)
	 (dolist (expr last-exprs)
	   (setq acc (nconc acc (recursive-transform expr))))
	 (setq last-exprs (set-difference (remove-duplicates acc :test #'equal)
					  all :test #'equal))))
    all))




(defun doit ()
   (funcall (compile-pattern-matcher
  '(&or (+ &commute (* $term1 $x)
                    (* &commute $term2 $x)
           &rest $addands)
        (+ &commute (* $x $term1)
	            (* &commute $term2 $x)
	   &rest $addands))
  '(
   `(+ (* (+ ,term1 ,term2) ,x) ,@addands)))
  '(+ y (* 7 x) 3 (* 9 x))))

(defun collect-transformers (groups)
  (when (symbolp groups)
     (setq groups (list groups)))
  (loop for (name tgroups transformer) in transformers
        when (intersection tgroups groups)
        collect (cons name transformer)))

(defun recursive-fixpoint (expr transformers)
  (if (atom expr)
      expr
      (progn
         (multiple-value-bind (subexprs imploder)
             (get-explosion-and-imploder expr)
             (setq expr (funcall imploder
                           (loop for sub in subexprs
                                 collect (recursive-fixpoint sub transformers)))))
         (loop for (name . transformer) in transformers
               for next-expr = (funcall transformer expr)
	       when next-expr do
;;                   (unless (eql (evaluate-expression expr '((N . 7)
;; 							   (i . 8)
;; 							   (j . 9)))
;; 			       (evaluate-expression next-expr '((N . 7)
;; 								(i . 8)
;; 								(j . 9))))
;; 		    (format t "ERROR ~A ~%~A ->~%~A~%" name expr next-expr))
                  (return (recursive-fixpoint next-expr transformers))
	       finally
                  (return expr)))))

(defun full-expand (expr)
  (recursive-fixpoint expr (collect-transformers 
    '(expanding constant-folding remove-extraneous))))

(defun full-folding (expr)
  (recursive-fixpoint expr (collect-transformers 
    '(folding constant-folding remove-extraneous))))

(defun full-transform (expr)
  (full-folding (full-expand expr)))

(defun evaluate-expression (expr &optional var-alist)
  (cond ((numberp expr)
	 expr)
	((symbolp expr)
	 (cdr (assoc expr var-alist)))
	((eq (car expr)
	     'sum)
	 (destructuring-bind (_ (index-var start end)
				body) expr
             (let ((start-index (evaluate-expression start var-alist))
		   (end-index (evaluate-expression end var-alist)))
;;                (format t "vars ~A~%" var-alist)
;;                (format t "start ~A = ~A~%" start start-index)
;;                (format t "end ~A = ~A~%" end end-index)
	       (assert (>= start-index 0))
	       (assert (>= end-index start-index))
	       (loop for i from start-index
		           upto end-index
		  sum (evaluate-expression body (cons (cons index-var i)
						      var-alist))))))
	(t (apply (symbol-function (car expr))
		  (loop for arg in (cdr expr)
		       collect (evaluate-expression arg var-alist))))))

(defun rec-make-sum (depth)
  (if (zerop depth)
      0
      (let ((index-var (intern (format nil "I~d" depth))))
      `(sum (,index-var j k)
	    (+ ,index-var (* ,depth (expt ,index-var 2))
               ,(rec-make-sum (1- depth)))))))

(print (full-transform goal-expression))