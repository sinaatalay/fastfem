# Writing Documentation

## $\LaTeX$ Equations with [KaTeX](https://katex.org)

To see all the available $\LaTeX$ commands, see [here](https://katex.org/docs/supported).

### Inline Equations

This is an example of an inline equation: $E=mc^2$.

```
This is an example of an inline equation: $E=mc^2$.
```

### Block Equations

This is an example of a block equation:

$$
E=mc^2
$$

```
$$
E=mc^2
$$
```

### Block Equations with Numbering

This is an example of a block equation with numbering:

$$
\begin{equation}
E=mc^2
\end{equation}
$$

```
$$
\begin{equation}
E=mc^2
\end{equation}
$$
```
### Cross-Referencing Block Equations

See [Equation 2](#eq-label) below.

!!! note
    The equation numbering is not automatic for cross-referencing, unlike figures and tables.

$$
\begin{equation}
E=mc^2
\end{equation}
$$
{ #eq:label }

```
See [Equation 2](#eq-label) below.

$$
\begin{equation}
E=mc^2
\end{equation}
$$
{ #eq:label }
```

## Citations with [BibTeX](http://www.bibtex.org/)

Update [`docs/assets/bibliography.bib`](https://github.com/sinaatalay/fastfem/blob/main/docs/assets/bibliography.bib) with your references.

This is an example of a citation[@Serway2014].

```
This is an example of a citation[@ Serway2014].
```

## Figures

### Figures without Captions

![](https://picsum.photos/200/300)

```markdown
![](https://picsum.photos/200/300)
```

### Figures with Captions

![This is the caption.](https://picsum.photos/200/300)

```markdown
![This is the caption.](https://picsum.photos/200/300)
```

### Cross-Referencing Figures with Captions

See [](#fig-label) below.

![This is the caption.](https://picsum.photos/200/300){ #fig-label }

```markdown
See [](#fig-label) below.

![This is the caption.](https://picsum.photos/200/300){ #fig-label }
```

## Diagrams with [Mermaid](https://mermaid.js.org)

```mermaid
graph TD;
    A-->B;
    A-->C;
    B-->D;
    C-->D;
```

``````markdown
```mermaid
graph TD;
    A-->B;
    A-->C;
    B-->D;
    C-->D;
```
``````

\bibliography