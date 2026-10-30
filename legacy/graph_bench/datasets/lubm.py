r"""
LUBM (Lehigh University Benchmark) dataset.

LUBM is the most widely used benchmark for RDF/OWL systems. It models
the university domain with departments, professors, students, courses,
and publications.

Ontology Classes:
    - University
    - Department
    - Professor (FullProfessor, AssociateProfessor, AssistantProfessor)
    - Lecturer
    - Student (GraduateStudent, UndergraduateStudent)
    - Course (GraduateCourse, UndergraduateCourse)
    - Publication
    - ResearchGroup

Properties:
    - memberOf, worksFor, teacherOf, takesCourse
    - advisor, publicationAuthor, headOf
    - degreeFrom, undergraduateDegreeFrom, mastersDegreeFrom, doctoralDegreeFrom
    - telephone, emailAddress, name

Scale:
    - 1 university ≈ 100K triples
    - Configurable number of universities

References:
    - http://swat.cse.lehigh.edu/projects/lubm/
    - Guo, Pan, Heflin: "LUBM: A Benchmark for OWL Knowledge Base Systems" (2005)

    from graph_bench.datasets.lubm import LUBM

    dataset = LUBM(universities=1)
    nodes, edges = dataset.generate(scale)  # LPG format
    triples = dataset.generate_rdf(scale)    # RDF triples
"""

import random
from typing import Any

from graph_bench.datasets.base import BaseDatasetLoader
from graph_bench.types import ScaleConfig

__all__ = ["LUBM"]

# LUBM namespace
LUBM_NS = "http://www.lehigh.edu/~zhp2/2004/0401/univ-bench.owl#"

# Typical counts per department (from LUBM spec)
DEPT_CONFIG = {
    "full_professors": (7, 10),
    "assoc_professors": (10, 14),
    "asst_professors": (8, 11),
    "lecturers": (5, 7),
    "grad_students": (10, 20),
    "undergrad_students": (50, 100),
    "grad_courses": (10, 20),
    "undergrad_courses": (15, 25),
    "research_groups": (1, 4),
}

# Course names
COURSE_SUBJECTS = [
    "Introduction to", "Advanced", "Fundamentals of", "Topics in",
    "Seminar on", "Research in", "Applied", "Theoretical",
]
COURSE_TOPICS = [
    "Computer Science", "Algorithms", "Data Structures", "Machine Learning",
    "Database Systems", "Networks", "Operating Systems", "Software Engineering",
    "Artificial Intelligence", "Computer Graphics", "Security", "Theory",
]

# Research areas
RESEARCH_AREAS = [
    "AI", "Systems", "Theory", "Graphics", "Networks", "Security",
    "Databases", "HCI", "Robotics", "NLP", "Vision", "Bioinformatics",
]


class LUBM(BaseDatasetLoader):
    """LUBM (Lehigh University Benchmark) dataset generator.

    Generates university domain data following LUBM ontology.
    Can output both LPG format (nodes, edges) and RDF triples.
    """

    def __init__(
        self,
        *,
        universities: int = 1,
        seed: int | None = None,
        departments_per_university: int = 15,
    ) -> None:
        """Initialize LUBM dataset generator.

        Args:
            universities: Number of universities to generate.
            seed: Random seed for reproducibility.
            departments_per_university: Departments per university.
        """
        self._universities = universities
        self._seed = seed
        self._departments = departments_per_university

    @property
    def name(self) -> str:
        return "lubm"

    def generate(
        self, scale: ScaleConfig
    ) -> tuple[list[dict[str, Any]], list[tuple[str, str, str, dict[str, Any]]]]:
        """Generate LUBM dataset in LPG format."""
        if self._seed is not None:
            random.seed(self._seed)

        nodes: list[dict[str, Any]] = []
        edges: list[tuple[str, str, str, dict[str, Any]]] = []

        node_count = 0
        edge_count = 0

        for u in range(self._universities):
            if node_count >= scale.nodes:
                break

            uni_id = f"uni_{u}"
            uni_name = f"University{u}"

            # University node
            nodes.append({
                "id": uni_id,
                "label": "University",
                "name": uni_name,
                "rdf_type": f"{LUBM_NS}University",
            })
            node_count += 1

            # Generate departments
            for d in range(self._departments):
                if node_count >= scale.nodes:
                    break

                dept_id = f"dept_{u}_{d}"
                dept_nodes, dept_edges = self._generate_department(
                    dept_id, uni_id, u, d, scale.nodes - node_count, scale.edges - edge_count
                )
                nodes.extend(dept_nodes)
                edges.extend(dept_edges)
                node_count += len(dept_nodes)
                edge_count += len(dept_edges)

        return nodes, edges

    def _generate_department(
        self,
        dept_id: str,
        uni_id: str,
        uni_idx: int,
        dept_idx: int,
        max_nodes: int,
        max_edges: int,
    ) -> tuple[list[dict[str, Any]], list[tuple[str, str, str, dict[str, Any]]]]:
        """Generate a single department with faculty, students, courses."""
        nodes: list[dict[str, Any]] = []
        edges: list[tuple[str, str, str, dict[str, Any]]] = []

        # Department node
        dept_name = f"Department{dept_idx}"
        nodes.append({
            "id": dept_id,
            "label": "Department",
            "name": dept_name,
            "rdf_type": f"{LUBM_NS}Department",
        })

        # Department -> University relationship
        edges.append((dept_id, uni_id, "subOrganizationOf", {}))

        # Track IDs for relationships
        faculty_ids: list[str] = []
        student_ids: list[str] = []
        grad_course_ids: list[str] = []
        undergrad_course_ids: list[str] = []

        # Generate faculty
        faculty_types = [
            ("FullProfessor", DEPT_CONFIG["full_professors"]),
            ("AssociateProfessor", DEPT_CONFIG["assoc_professors"]),
            ("AssistantProfessor", DEPT_CONFIG["asst_professors"]),
            ("Lecturer", DEPT_CONFIG["lecturers"]),
        ]

        for faculty_type, (min_count, max_count) in faculty_types:
            count = random.randint(min_count, max_count)
            for i in range(count):
                if len(nodes) >= max_nodes:
                    break

                fac_id = f"{dept_id}_{faculty_type.lower()}_{i}"
                fac_name = f"{faculty_type}{i}_{dept_name}"

                nodes.append({
                    "id": fac_id,
                    "label": faculty_type,
                    "name": fac_name,
                    "emailAddress": f"{fac_id}@{dept_name}.edu",
                    "telephone": f"+1-{random.randint(100, 999)}-{random.randint(1000, 9999)}",
                    "researchInterest": random.choice(RESEARCH_AREAS),
                    "rdf_type": f"{LUBM_NS}{faculty_type}",
                })
                faculty_ids.append(fac_id)

                # worksFor relationship
                if len(edges) < max_edges:
                    edges.append((fac_id, dept_id, "worksFor", {}))

                # degreeFrom some university (could be same or different)
                if len(edges) < max_edges and random.random() > 0.3:
                    degree_uni = f"uni_{random.randint(0, self._universities - 1)}"
                    edges.append((fac_id, degree_uni, "doctoralDegreeFrom", {}))

        # Assign department head
        if faculty_ids and len(edges) < max_edges:
            head_id = random.choice(faculty_ids)
            edges.append((head_id, dept_id, "headOf", {}))

        # Generate graduate students
        grad_count = random.randint(*DEPT_CONFIG["grad_students"])
        for i in range(grad_count):
            if len(nodes) >= max_nodes:
                break

            student_id = f"{dept_id}_gradstudent_{i}"
            nodes.append({
                "id": student_id,
                "label": "GraduateStudent",
                "name": f"GraduateStudent{i}_{dept_name}",
                "emailAddress": f"{student_id}@{dept_name}.edu",
                "rdf_type": f"{LUBM_NS}GraduateStudent",
            })
            student_ids.append(student_id)

            # memberOf relationship
            if len(edges) < max_edges:
                edges.append((student_id, dept_id, "memberOf", {}))

            # advisor relationship (to a faculty member)
            if faculty_ids and len(edges) < max_edges:
                advisor_id = random.choice(faculty_ids)
                edges.append((student_id, advisor_id, "advisor", {}))

            # undergraduateDegreeFrom
            if len(edges) < max_edges and random.random() > 0.5:
                degree_uni = f"uni_{random.randint(0, self._universities - 1)}"
                edges.append((student_id, degree_uni, "undergraduateDegreeFrom", {}))

        # Generate undergraduate students
        undergrad_count = random.randint(*DEPT_CONFIG["undergrad_students"])
        for i in range(undergrad_count):
            if len(nodes) >= max_nodes:
                break

            student_id = f"{dept_id}_undergradstudent_{i}"
            nodes.append({
                "id": student_id,
                "label": "UndergraduateStudent",
                "name": f"UndergraduateStudent{i}_{dept_name}",
                "emailAddress": f"{student_id}@{dept_name}.edu",
                "rdf_type": f"{LUBM_NS}UndergraduateStudent",
            })
            student_ids.append(student_id)

            # memberOf relationship
            if len(edges) < max_edges:
                edges.append((student_id, dept_id, "memberOf", {}))

        # Generate graduate courses
        grad_course_count = random.randint(*DEPT_CONFIG["grad_courses"])
        for i in range(grad_course_count):
            if len(nodes) >= max_nodes:
                break

            course_id = f"{dept_id}_gradcourse_{i}"
            course_name = f"{random.choice(COURSE_SUBJECTS)} {random.choice(COURSE_TOPICS)}"
            nodes.append({
                "id": course_id,
                "label": "GraduateCourse",
                "name": course_name,
                "rdf_type": f"{LUBM_NS}GraduateCourse",
            })
            grad_course_ids.append(course_id)

            # teacherOf relationship
            if faculty_ids and len(edges) < max_edges:
                teacher_id = random.choice(faculty_ids)
                edges.append((teacher_id, course_id, "teacherOf", {}))

        # Generate undergraduate courses
        undergrad_course_count = random.randint(*DEPT_CONFIG["undergrad_courses"])
        for i in range(undergrad_course_count):
            if len(nodes) >= max_nodes:
                break

            course_id = f"{dept_id}_undergradcourse_{i}"
            course_name = f"{random.choice(COURSE_SUBJECTS)} {random.choice(COURSE_TOPICS)}"
            nodes.append({
                "id": course_id,
                "label": "UndergraduateCourse",
                "name": course_name,
                "rdf_type": f"{LUBM_NS}UndergraduateCourse",
            })
            undergrad_course_ids.append(course_id)

            # teacherOf relationship
            if faculty_ids and len(edges) < max_edges:
                teacher_id = random.choice(faculty_ids)
                edges.append((teacher_id, course_id, "teacherOf", {}))

        # Students take courses
        all_courses = grad_course_ids + undergrad_course_ids
        for student_id in student_ids:
            if len(edges) >= max_edges:
                break
            # Each student takes 1-4 courses
            course_count = random.randint(1, 4)
            for course_id in random.sample(all_courses, min(course_count, len(all_courses))):
                if len(edges) >= max_edges:
                    break
                edges.append((student_id, course_id, "takesCourse", {}))

        # Generate research groups
        rg_count = random.randint(*DEPT_CONFIG["research_groups"])
        for i in range(rg_count):
            if len(nodes) >= max_nodes:
                break

            rg_id = f"{dept_id}_researchgroup_{i}"
            nodes.append({
                "id": rg_id,
                "label": "ResearchGroup",
                "name": f"ResearchGroup{i}_{dept_name}",
                "researchArea": random.choice(RESEARCH_AREAS),
                "rdf_type": f"{LUBM_NS}ResearchGroup",
            })

            # subOrganizationOf department
            if len(edges) < max_edges:
                edges.append((rg_id, dept_id, "subOrganizationOf", {}))

        # Generate publications
        for fac_id in faculty_ids:
            if len(nodes) >= max_nodes:
                break
            # Each faculty has 0-5 publications
            pub_count = random.randint(0, 5)
            for i in range(pub_count):
                if len(nodes) >= max_nodes:
                    break

                pub_id = f"{fac_id}_pub_{i}"
                nodes.append({
                    "id": pub_id,
                    "label": "Publication",
                    "name": f"Publication{i} by {fac_id}",
                    "rdf_type": f"{LUBM_NS}Publication",
                })

                # publicationAuthor relationship
                if len(edges) < max_edges:
                    edges.append((pub_id, fac_id, "publicationAuthor", {}))

                # Sometimes has co-authors
                if faculty_ids and len(edges) < max_edges and random.random() > 0.6:
                    coauthor = random.choice(faculty_ids)
                    if coauthor != fac_id:
                        edges.append((pub_id, coauthor, "publicationAuthor", {}))

        return nodes, edges

    def generate_rdf(self, scale: ScaleConfig) -> list[tuple[str, str, str]]:
        """Generate LUBM dataset as RDF triples (subject, predicate, object).

        This format is suitable for RDF/SPARQL benchmarking.

        Returns:
            List of (subject, predicate, object) triples.
        """
        nodes, edges = self.generate(scale)

        triples: list[tuple[str, str, str]] = []

        # Convert nodes to RDF triples
        for node in nodes:
            subject = f"{LUBM_NS}{node['id']}"

            # rdf:type triple
            if "rdf_type" in node:
                triples.append((subject, "http://www.w3.org/1999/02/22-rdf-syntax-ns#type", node["rdf_type"]))

            # Literal properties
            for key, value in node.items():
                if key in ("id", "label", "rdf_type") or value is None:
                    continue
                predicate = f"{LUBM_NS}{key}"
                obj = f'"{value}"' if isinstance(value, str) else str(value)
                triples.append((subject, predicate, obj))

        # Convert edges to RDF triples
        for src, tgt, rel_type, props in edges:
            subject = f"{LUBM_NS}{src}"
            predicate = f"{LUBM_NS}{rel_type}"
            obj = f"{LUBM_NS}{tgt}"
            triples.append((subject, predicate, obj))

        return triples
