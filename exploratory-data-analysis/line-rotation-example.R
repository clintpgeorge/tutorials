library(MASS)

draw.segment <- function(p1,
                         p2,
                         color,
                         width = 3,
                         lty = 1) {
  segments(p1[1],
           p1[2],
           p2[1],
           p2[2],
           col = color,
           lwd = width,
           lty = lty)
}


# draw a segment (initially at 45 degrees)
p.init <- matrix(c(0, 0), 2, 1)
p.end  <- matrix(c(2, 2), 2, 1)
draw.segment(p.init, p.end, "red")

# rotation example
theta <- pi / 4  # rotate more 45º degrees
rotate.matrix <- matrix(c(cos(theta), sin(theta),-sin(theta), cos(theta)), 2, 2)

p1.init <- rotate.matrix %*% p.init
p1.end  <- rotate.matrix %*% p.end


plot(
  NULL,
  xlim = c(-.2, 3),
  ylim = c(-.2, 3),
  ylab = "y",
  xlab = "x",
  cex = 3
)
draw.segment(p1.init, p1.end, "blue")
legend(
  "bottomright",
  c("Initial Line", "Rotated 45%"),
  col = c("red", "blue"),
  lwd = 1
)


# scaling example
scalingx <- 0.5  # scale both axis by 50%
scalingy <- 0.5
scale.matrix <- matrix(c(scalingx,    0,
                         0,     scalingy), 2, 2)

p2.init <- scale.matrix %*% p.init
p2.end  <- scale.matrix %*% p.end

plot(
  NULL,
  xlim = c(-.2, 3),
  ylim = c(-.2, 3),
  ylab = "y",
  xlab = "x",
  cex = 3
)
draw.segment(p2.init, p2.end, "green", lty = 2)
legend(
  "bottomright",
  c("Initial Line", "Rotated 45%", "Scaled 50%"),
  col = c("red", "blue", "green"),
  lwd = 1
)